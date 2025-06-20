import numpy as np
from preprocessing import *
import matplotlib.pyplot as plt
import time 

field_size = 100
# field_resolution = (field_size, field_size)
field_range = np.array([[0, field_size], [0, field_size]])
# num_sensors = 10
AGENT_COMM_RANGE = 20
            
# num_sensors = number_of_sensors(AGENT_COMM_RANGE, field_size)
num_sensors = 149 # Set a fixed number of sensors for testing (look for median rule on odd/even number of sensors)

print(f"Number of sensors: {num_sensors}")
print("probability:", 1-1/num_sensors**2)

agent_locations = np.array([0.9*np.random.uniform(field_range[0,0], field_range[0,1], num_sensors),
                            0.9*np.random.uniform(field_range[1,0], field_range[1,1], num_sensors)])

sensor_readings = np.random.uniform(10, 40, num_sensors) + np.random.normal(0, 0.1, num_sensors)  # Simulated sensor readings with noise


print(f"Sensor readings shape: {sensor_readings}")
true_median_temp = np.median(sensor_readings)
print(f"True median temperature: {true_median_temp:.2f} °C")
# np.random.seed(42)  # For reproducibility
# converged = False

# avg_temperature = np.mean(temperature_measurements[0])
# print(f"Average temperature: {avg_temperature:.2f} °C")

adjacency_matrix = calculate_adjacency_matrix(agent_locations, AGENT_COMM_RANGE)
if not is_graph_connected(adjacency_matrix):
    print("Warning: The graph is not connected.")
else:
    print("The graph is connected.")
visualize_graphs(agent_locations, adjacency_matrix)


def pdmm_median_algorithm(adjacency_matrix, initial_vals, num_iterations = 50000, c = 1.0, tole = 1e-6):
    
    initial_vals = np.array(initial_vals, dtype=float)  # Ensure initial_vals is a float array
    current_vals = initial_vals.copy()
    # n = len(current_vals)
    n = adjacency_matrix.shape[0]
    # estimated_vals = [current_vals.copy()]
    x = initial_vals.copy()
    estimated_vals = [x.copy()]
    median_true_vals = np.median(initial_vals)
    
    z = dict()
    # y = dict()
    
    num_edges = 0
    
    # x_new = np.zeros(n)  # Initialize x as a zero vector of size n

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0 and i != j:
                # connected_nodes.append((i, j))
                num_edges += 1
                z[(i, j)] = 0

                # y[(i, j)] = initial_vals[i] 

    transmission_count = 0
    tx_counts = [transmission_count]
    print(f"Number of edges in the graph: {num_edges}")
    # print(f"Number of connected nodes: {len(connected_nodes)}")

    for it in range(num_iterations):

        # x_new = np.zeros(n)  # Initialize x_new as a zero vector of size n
        for i in range(n):
            # x_new[i] = current_vals[i]
            x[i] = current_vals[i]  # Initialize x with the current values
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            # num_neighbors = len(neighbors)
            # print("Neighbors: ", num_neighbors)
            s = 0
            for j in neighbors:
            # Update x based on z
                A = 1 if i < j else -1
                # x_new[i] -=  A*z[(i, j)] 
                s += A*z[(i, j)]

            if (-1-s)/(c*len(neighbors)) > initial_vals[i]:

                x[i] = (-1-s)/(c*len(neighbors))

            elif (1-s)/(c*len(neighbors)) < initial_vals[i]:

                x[i] = (1-s)/(c*len(neighbors))
            
            else:

                x[i] = initial_vals[i]

        # x = x_new.copy()  # Update x_new with the new values
        z_prev = z.copy()
        for i in range(n):
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]
            for j in neighbors:
                A = 1 if i < j else -1
                z[(j, i)] = 0.5*z_prev[(j, i)] + 0.5*(z_prev[(i, j)] + 2*c*A*x[i])

                transmission_count += 1  # Count the transmission

        tx_counts.append(transmission_count)
        estimated_vals.append(x.copy())
        if np.linalg.norm(x - true_median_temp*np.ones(n)) < tole :
            # print(f"Convergence reached at iteration {it}.")
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            
            return estimated_vals, tx_counts # Return the estimates and the number of iterations
        
        if it%1000 ==0:
            
            print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")

           
    print("Maximum iterations reached without convergence.")
    return estimated_vals, tx_counts


def pdmm_median_algorithm_broadcast(adjacency_matrix, initial_vals, num_iterations = 50000, c = 1.0, tole = 1e-6):
    
    initial_vals = np.array(initial_vals, dtype=float)  # Ensure initial_vals is a float array
    current_vals = initial_vals.copy()
    # n = len(current_vals)
    n = adjacency_matrix.shape[0]
    # estimated_vals = [current_vals.copy()]
    x = initial_vals.copy()
    estimated_vals = [x.copy()]
    median_true_vals = np.median(initial_vals)
    
    z = dict()
    # y = dict()
    
    num_edges = 0
    
    # x_new = np.zeros(n)  # Initialize x as a zero vector of size n

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0 and i != j:
                # connected_nodes.append((i, j))
                num_edges += 1
                z[(i, j)] = 0

                # y[(i, j)] = initial_vals[i] 

    transmission_count = 0
    tx_counts = [transmission_count]
    print(f"Number of edges in the graph: {num_edges}")
    # print(f"Number of connected nodes: {len(connected_nodes)}")

    for it in range(num_iterations):

        # x_new = np.zeros(n)  # Initialize x_new as a zero vector of size n
        for i in range(n):
            # x_new[i] = current_vals[i]
            x[i] = current_vals[i]  # Initialize x with the current values
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            # num_neighbors = len(neighbors)
            # print("Neighbors: ", num_neighbors)
            s = 0
            for j in neighbors:
            # Update x based on z
                A = 1 if i < j else -1
                # x_new[i] -=  A*z[(i, j)] 
                s += A*z[(i, j)]

            if (-1-s)/(c*len(neighbors)) > initial_vals[i]:

                x[i] = (-1-s)/(c*len(neighbors))

            elif (1-s)/(c*len(neighbors)) < initial_vals[i]:

                x[i] = (1-s)/(c*len(neighbors))
            
            else:

                x[i] = initial_vals[i]

        # x = x_new.copy()  # Update x_new with the new values
        z_prev = z.copy()
        for i in range(n):
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]
            for j in neighbors:
                A = 1 if i < j else -1
                z[(j, i)] = 0.5*z_prev[(j, i)] + 0.5*(z_prev[(i, j)] + 2*c*A*x[i])

            transmission_count += 1  # Count the transmission

        tx_counts.append(transmission_count)
        estimated_vals.append(x.copy())
        if np.linalg.norm(x - true_median_temp*np.ones(n) ) < tole and 'converged' not in locals():
            # print(f"Convergence reached at iteration {it}.")
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            converged = True
            return estimated_vals, tx_counts # Return the estimates and the number of iterations
        
        if it%1000 ==0:
            
            print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")

           
    print("Maximum iterations reached without convergence.")
    return estimated_vals, tx_counts

def plot_PDMM_results(estimates_list, tx_counts_list, true_median_temp, labels):
    plt.figure(figsize=(10, 6))
    
    for estimates, tx_counts, label in zip(estimates_list, tx_counts_list, labels):
        errors = [np.linalg.norm(x - true_median_temp)**2 for x in estimates]
        plt.semilogy(tx_counts, errors, label=label)
    
    plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')
    plt.xlabel('Transmission Count')
    plt.ylabel('L2 Norm of Error (Squared)')
    plt.title('Convergence of PDMM Median Algorithm')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the PDMM algorithm for median estimation
time_start = time.time()
estimates_uni, tx_counts_uni = pdmm_median_algorithm(adjacency_matrix, sensor_readings)
time_end = time.time()
print(f"PDMM Median Algorithm (Unicast) completed in {time_end - time_start:.2f} seconds.")

# metrics_pdmm = convergence_summary(sensor_readings, estimates_uni, tx_counts_uni)

# print(f"PDMM Algorithm (Unicast) Metrics:")
# print(f"Iterations to converge: {metrics_pdmm['iterations_to_converge']}")
# print(f"Transmissions to converge: {metrics_pdmm['transmissions_to_converge']}")
# print(f"Final error: {metrics_pdmm['final_error']:.2e}")


time_start_bro = time.time()
estimates_broadcast, tx_counts_bro = pdmm_median_algorithm_broadcast(adjacency_matrix, sensor_readings)
time_end_bro = time.time()
print(f"PDMM Median Algorithm (Broadcast) completed in {time_end_bro - time_start_bro:.2f} seconds.")

# metrics_pdmm_bro = convergence_summary(sensor_readings, estimates_broadcast, tx_counts_bro)

# print(f"PDMM Algorithm (Unicast) Metrics:")
# print(f"Iterations to converge: {metrics_pdmm_bro['iterations_to_converge']}")
# print(f"Transmissions to converge: {metrics_pdmm_bro['transmissions_to_converge']}")
# print(f"Final error: {metrics_pdmm_bro['final_error']:.2e}")

# errors_unicast = [np.linalg.norm(x - true_median_temp)**2 for x in estimates_uni]
# errors_broadcast = [np.linalg.norm(x - true_median_temp)**2 for x in estimates_broadcast]

# # Plot both curves on the same figure
# plt.figure(figsize=(10, 6))
# plt.semilogy(tx_counts_uni, errors_unicast, label="PDMM Median (Unicast)")
# plt.semilogy(tx_counts_bro, errors_broadcast, label="PDMM Median (Broadcast)")
# plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')
# plt.xlabel("Transmission Count")
# plt.ylabel("L2 Norm of Error (Squared)")
# plt.title("Convergence of PDMM Median Algorithm (Unicast vs Broadcast)")
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()


plot_PDMM_results(
    estimates_list=[estimates_uni, estimates_broadcast],
    tx_counts_list=[tx_counts_uni, tx_counts_bro],
    true_median_temp=true_median_temp,
    labels=["PDMM Median (Unicast)", "PDMM Median (Broadcast)"]
)
# Plot the results
# plot_PDMM_results(estimates, tx_counts , true_median_temp) 

        
                

                

     



