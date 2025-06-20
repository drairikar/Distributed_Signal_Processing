import numpy as np
from preprocessing import *
import matplotlib.pyplot as plt
import random

field_size = 100
# field_resolution = (field_size, field_size)
field_range = np.array([[0, field_size], [0, field_size]])
# num_sensors = 10
AGENT_COMM_RANGE = 40
# PACKET_LOSS_RATE
            
num_sensors = number_of_sensors(AGENT_COMM_RANGE, field_size)
print(f"Number of sensors: {num_sensors}")
print("probability:", 1-1/num_sensors**2)

agent_locations = np.array([0.9*np.random.uniform(field_range[0,0], field_range[0,1], num_sensors),
                            0.9*np.random.uniform(field_range[1,0], field_range[1,1], num_sensors)])

sensor_readings = np.random.uniform(10, 40, num_sensors) + np.random.normal(0, 0.1, num_sensors)  # Simulated sensor readings with noise


print(f"Sensor readings shape: {sensor_readings}")
true_avg_temp = np.mean(sensor_readings)

print(f"True average temperature: {true_avg_temp:.2f} °C")

true_median_temp = np.median(sensor_readings)
print(f"True median temperature: {true_median_temp:.2f} °C")

# avg_temperature = np.mean(temperature_measurements[0])
# print(f"Average temperature: {avg_temperature:.2f} °C")

adjacency_matrix = calculate_adjacency_matrix(agent_locations, AGENT_COMM_RANGE)
if not is_graph_connected(adjacency_matrix):
    print("Warning: The graph is not connected.")
else:
    print("The graph is connected.")
visualize_graphs(agent_locations, adjacency_matrix)


def pdmm_algorithm(adjacency_matrix, initial_vals, num_iterations = 50000, c = 1.0, tole = 1e-6, packet_loss_rate = 0):
    
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
    
    # x_new = np.zeros(n)  # Initialize x as a zero vector of size n

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0 and i != j:
                # connected_nodes.append((i, j))
                num_edges += 1
                z[(i, j)] = 0
                y[(i, j)] = 0

    transmission_count = 0
    tx_counts = [transmission_count]
    # tx_counts_fail = [0]
    total_lost = 0
    # # rng = random.Random()

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
            
        x = x_new

        for i in range(n):
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                A = 1 if i < j else -1
                y[(i, j)] = z[(i , j)] + 2*c*A*x[i]

        # x = x_new.copy()

        # Simulate packet loss
        for i in range(n):
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors

            for j in neighbors:
                transmission_count += 1  # Count the transmission

                if np.random.rand() < packet_loss_rate:
                    total_lost += 1
                    continue  # Simulate packet loss
                        
                A = 1 if i < j else -1
                z[(j, i)] = y[(i, j)]
                            
                
        # Check convergence
        estimated_vals.append(x.copy())
        tx_counts.append(transmission_count)
        # tx_counts_fail.append(total_lost)

        if np.linalg.norm(x - avg_true_vals*np.ones(n)) < tole:
        # if np.linalg.norm(x - median_true_vals*np.ones(n)) < tole:
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            print(f"Total packets lost: {total_lost}")
            return estimated_vals, tx_counts, total_lost # Return the estimates and the number of iterations
        
        if it%100 ==0:
            print(f"Iteration {it}: Estimated average: {np.mean(x)}, True average: {np.mean(avg_true_vals)}, error: {np.linalg.norm(x - avg_true_vals*np.ones(n))}, Transmission count: {transmission_count}")
            # print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")

        
    print("Maximum iterations reached without convergence.")
        
    return estimated_vals, tx_counts, total_lost

def pdmm_median_algorithm(adjacency_matrix, initial_vals, num_iterations = 50000, c=1.0, tole=1e-6, packet_loss_rate=0):
    
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
    
    x_new = np.zeros(n)  # Initialize x as a zero vector of size n

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0 and i != j:
                # connected_nodes.append((i, j))
                num_edges += 1
                z[(i, j)] = 0
                # y[(i, j)] = initial_vals[i] 

    transmission_count = 0
    total_lost = 0
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
            s = 0
            for j in neighbors:
            # Update x based on z
                A = 1 if i < j else -1
                # x_new[i] -=  A*z[(i, j)] 
                s += A*z[(i, j)]

            if (-1-s)/(c*len(neighbors)) > initial_vals[i]:

                x_new[i] = (-1-s)/(c*len(neighbors))

            elif (1-s)/(c*len(neighbors)) < initial_vals[i]:

                x_new[i] = (1-s)/(c*len(neighbors))
            
            else:

                x_new[i] = initial_vals[i]

        # Update x with the new values
            x = x_new.copy()
            z_prev = z.copy()

        # for i in range(n):
        #     neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]
            for j in neighbors:
                transmission_count += 1
                # Simulate packet loss
                # print(np.random.rand(), packet_loss_rate)
                if np.random.rand() < packet_loss_rate:
                    total_lost += 1
                    continue
                A = 1 if i < j else -1
                z[(j,i)] = 0.5*z_prev[(j, i)] + 0.5*(z_prev[(i, j)] + 2*c*A*x[i])

        
        tx_counts.append(transmission_count)

        estimated_vals.append(x.copy())
        if np.linalg.norm(x - true_median_temp*np.ones(n)) < tole:
            # print(f"Convergence reached at iteration {it}.")
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            return estimated_vals, tx_counts, total_lost # Return the estimates and the number of iterations
        
        if it%100 ==0:
            
            print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")

           
    print("Maximum iterations reached without convergence.")
    return estimated_vals, tx_counts, total_lost


def packet_loss_plots(adjacency_matrix, initial_vals, num_iterations = 50000, tole = 1e-6):
    packet_loss_rates = [0.0, 0.25, 0.5, 0.75]
    # true_avg_temp = np.mean(initial_vals)
    true_median_temp = np.median(initial_vals)
    plt.figure(figsize=(8, 5))

    for loss_rate in packet_loss_rates:
        print(f"\nRunning PDMM with packet loss rate = {loss_rate}")
        estimates, tx_counts, failed = pdmm_median_algorithm(
            adjacency_matrix, initial_vals,
            num_iterations=num_iterations,
            tole=tole,
            packet_loss_rate=loss_rate
        )

        if not estimates:
            print(f"No estimates returned for loss rate = {loss_rate}")
            continue

        # errors = [np.linalg.norm(x - true_median_temp)**2 for x in estimates]
        errors = [np.linalg.norm(x - true_median_temp)**2 for x in estimates]
        label = f"Loss = {int(loss_rate*100)}%, Failed = {failed}"
        plt.semilogy(tx_counts, errors, label=label)

    plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')
    plt.xlabel('Transmission Count')
    plt.ylabel('L2 Norm of Error')
    plt.title('PDMM Convergence with Varying Packet Loss Rates')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# estimates, txcounts, failed = pdmm_algorithm(adjacency_matrix, sensor_readings)
# packet_loss_plots(adjacency_matrix, sensor_readings, num_iterations=50000, tole=1e-6)

# Run the PDMM algorithm for median estimation
estimates_median, tx_counts_median, total_lost = pdmm_median_algorithm(adjacency_matrix, sensor_readings)
packet_loss_plots(adjacency_matrix, sensor_readings, num_iterations=50000, tole=1e-6)

