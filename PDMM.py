import numpy as np
from preprocessing import *
import matplotlib.pyplot as plt

field_size = 100
# field_resolution = (field_size, field_size)
field_range = np.array([[0, field_size], [0, field_size]])
# num_sensors = 10
AGENT_COMM_RANGE = 40
            
num_sensors = number_of_sensors(AGENT_COMM_RANGE, field_size)
print(f"Number of sensors: {num_sensors}")
print("probability:", 1-1/num_sensors**2)

agent_locations = np.array([0.9*np.random.uniform(field_range[0,0], field_range[0,1], num_sensors),
                            0.9*np.random.uniform(field_range[1,0], field_range[1,1], num_sensors)])

sensor_readings = np.random.uniform(10, 40, num_sensors) + np.random.normal(0, 0.1, num_sensors)  # Simulated sensor readings with noise


print(f"Sensor readings shape: {sensor_readings}")
true_avg_temp = np.mean(sensor_readings)

print(f"True average temperature: {true_avg_temp:.2f} °C")

# avg_temperature = np.mean(temperature_measurements[0])
# print(f"Average temperature: {avg_temperature:.2f} °C")

adjacency_matrix = calculate_adjacency_matrix(agent_locations, AGENT_COMM_RANGE)
if not is_graph_connected(adjacency_matrix):
    print("Warning: The graph is not connected.")
else:
    print("The graph is connected.")
visualize_graphs(agent_locations, adjacency_matrix)


def pdmm_algorithm(adjacency_matrix, initial_vals, num_iterations = 50000, c = 1.0, tole = 1e-4):
    
    initial_vals = np.array(initial_vals, dtype=float)  # Ensure initial_vals is a float array
    current_vals = initial_vals.copy()
    # n = len(current_vals)
    n = adjacency_matrix.shape[0]
    # estimated_vals = [current_vals.copy()]
    x = initial_vals.copy()
    estimated_vals = [x.copy()]
    avg_true_vals = np.mean(initial_vals)
    # median_true_vals = np.median(initial_vals)
    z = dict()
    y = dict()
    
    num_edges = 0
    
    x_new = np.zeros(n)  # Initialize x as a zero vector of size n

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0 and i != j:
                # connected_nodes.append((i, j))
                num_edges += 1
                z[(i, j)] = initial_vals[i]
                y[(i, j)] = initial_vals[i] 

    transmission_count = 0
    tx_counts = [transmission_count]
    print(f"Number of edges in the graph: {num_edges}")
    # print(f"Number of connected nodes: {len(connected_nodes)}")

    for it in range(num_iterations):
        # x_prev = x.copy()  # Store the previous values of x
        x_new = np.zeros(n)  # Initialize x_new as a zero vector of size n
        for i in range(n):
            x_new[i] = current_vals[i]  # Update x with the current values
    
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            # num_neighbors = len(neighbors)
            # print("Neighbors: ", num_neighbors)
            for j in neighbors:
            # Update x based on z
                A = 1 if i < j else -1
                x_new[i] -=  A*z[(i, j)] 
            x_new[i] /= (1 + c*len(neighbors))  # Normalize by the number of neighbors
        # Update x with the new values
        x = x_new.copy()

        for i in range(n):
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                A = 1 if i < j else -1
                y[(i, j)] = z[(i , j)] + 2*c*A*x[i]

        for i in range(n):
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                A = 1 if i < j else -1
                z[(j, i)] = y[(i, j)] 
                transmission_count += 1  # Count the transmission
        # Check convergence
        estimated_vals.append(x.copy())
        tx_counts.append(transmission_count)
        if np.linalg.norm(x - avg_true_vals*np.ones(n)) < tole:
        # if np.linalg.norm(x - median_true_vals*np.ones(n)) < tole:
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            return estimated_vals, tx_counts # Return the estimates and the number of iterations
        
        if it%100 ==0:
            print(f"Iteration {it}: Estimated average: {np.mean(x)}, True average: {np.mean(avg_true_vals)}, error: {np.linalg.norm(x - avg_true_vals*np.ones(n))}, Transmission count: {transmission_count}")
            # print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")


    print("Maximum iterations reached without convergence.")
        
    return estimated_vals, tx_counts

def pdmm_algorithm_broadcast(adjacency_matrix, initial_vals, num_iterations = 50000, c = 1.0, tole = 1e-6):

    initial_vals = np.array(initial_vals, dtype=float)  # Ensure initial_vals is a float array
    current_vals = initial_vals.copy()
    # n = len(current_vals)
    n = adjacency_matrix.shape[0]
    # estimated_vals = [current_vals.copy()]
    x = initial_vals.copy()
    estimated_vals = [x.copy()]
    avg_true_vals = np.mean(initial_vals)
    # median_true_vals = np.median(initial_vals)
    z = dict()
    y = dict()
    
    num_edges = 0
    
    x_new = np.zeros(n)  # Initialize x as a zero vector of size n

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0 and i != j:
                # connected_nodes.append((i, j))
                num_edges += 1
                z[(i, j)] = initial_vals[i]
                y[(i, j)] = initial_vals[i] 

    transmission_count = 0
    tx_counts = [transmission_count]
    print(f"Number of edges in the graph: {num_edges}")
    # print(f"Number of connected nodes: {len(connected_nodes)}")

    for it in range(num_iterations):
        # x_prev = x.copy()  # Store the previous values of x
        x_new = np.zeros(n)  # Initialize x_new as a zero vector of size n
        for i in range(n):
            x_new[i] = current_vals[i]  # Update x with the current values
    
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            print("Neighbors: ", neighbors)
            # num_neighbors = len(neighbors)
            # print("Neighbors: ", num_neighbors)
            for j in neighbors:
            # Update x based on z
                # A = 1 if i < j else -1
                x_new[i] -=  z[(i, j)] 
            x_new[i] /= (1 + c*len(neighbors))  # Normalize by the number of neighbors
        # Update x with the new values
        x = x_new.copy()

        for i in range(n):
            
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                # A = 1 if i < j else -1
                y[(i, j)] = z[(i , j)] + 2*c*x[i]

        for i in range(n):
            
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                # A = 1 if i < j else -1
                z[(j, i)] = -y[(i, j)] 
            transmission_count += 1  # Count the transmission

        # Check convergence
        estimated_vals.append(x.copy())
        tx_counts.append(transmission_count)
        if np.linalg.norm(x - avg_true_vals*np.ones(n)) < tole:
        # if np.linalg.norm(x - median_true_vals*np.ones(n)) < tole:
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            return estimated_vals, tx_counts # Return the estimates and the number of iterations
        
        if it%100 ==0:
            print(f"Iteration {it}: Estimated average: {np.mean(x)}, True average: {np.mean(avg_true_vals)}, error: {np.linalg.norm(x - avg_true_vals*np.ones(n))}, Transmission count: {transmission_count}")
            # print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")


    print("Maximum iterations reached without convergence.")
        
    return estimated_vals, tx_counts

    
def plot_PDMM_results(estimates_unicast,tx_counts_unicast,estimates_broadcast,tx_counts_broadcast, true_avg_temp):
    ##Plot 
    # if not len(estimates):
    #     print("No estimated values to plot.")
    #     return
    
    # print("Number of transmissions:", tx_counts)
    
    errors_unicast = [np.linalg.norm(x - true_avg_temp)**2 for x in estimates_unicast]
    # print("Errors:", len(errors))
    plt.semilogy(tx_counts_unicast, errors_unicast, label="PDMM Estimates (Unicast)")

    errors_broadcast = [np.linalg.norm(x - true_avg_temp)**2 for x in estimates_broadcast]
    plt.semilogy(tx_counts_broadcast, errors_broadcast, label="PDMM Estimates (Broadcast)")
    # plt.yscale('log')
    plt.xlabel('Transmission Count')
    plt.ylabel('L2 norm of Error')
    plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')
    plt.title('Convergence of PDMM Algorithm ')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()

##Plot both PDMM and PDMM broadcast results

estimates, txcounts = pdmm_algorithm(adjacency_matrix, sensor_readings)
# plot_PDMM_results(estimates, txcounts, true_avg_temp) 
estimates_broadcast, txcounts_broadcast = pdmm_algorithm_broadcast(adjacency_matrix, sensor_readings)
plot_PDMM_results(estimates, txcounts, estimates_broadcast ,txcounts_broadcast, true_avg_temp) 

    




    

