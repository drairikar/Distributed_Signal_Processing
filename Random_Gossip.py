from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import time

field_size = 100
# field_resolution = (field_size, field_size)
field_range = np.array([[0, field_size], [0, field_size]])
# num_sensors = 10
AGENT_COMM_RANGE = 20
            
# num_sensors = number_of_sensors(AGENT_COMM_RANGE, field_size)
num_sensors = 150
print(f"Number of sensors: {num_sensors}")
print("probability:", 1-1/num_sensors**2)

agent_locations = np.array([0.9*np.random.uniform(field_range[0,0], field_range[0,1], num_sensors),
                            0.9*np.random.uniform(field_range[1,0], field_range[1,1], num_sensors)])

##get sensor measurements
# temperature_measurements = temperature_generator(field_size=field_size, field_resolution=(field_size, field_size), min_temp=10, max_temp=40)
# temp_temperature_measurements = temperature_measurements[0]  # Get the temperature field
# sensor_readings = []
# field_res_x, field_res_y = temp_temperature_measurements.shape
# x_p = np.linspace(field_range[0,0], field_range[0,1], field_res_x)
# y_p = np.linspace(field_range[1,0], field_range[1,1], field_res_y)
# interp = RegularGridInterpolator((x_p, y_p), temp_temperature_measurements)

# pts_to_sample = agent_locations.T  # Transpose to get points in the format (num_sensors, 2)
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

##Randomized Gossip Algorithm
##selecting a random node i and a random neighbor j having probability p_ij.
##they set their temperature to the average of their own temperature and the neighbor's temperature.

##Optimizing the gossip algorithm to ensure convergence
##x(k) = W(k)x(k-1), Wij = I-0.5(e_i-e_j)(e_i-e_j)^T, W_bar = 1/n P_ij W_ij 

def calculate_P_matrix(adjacency_matrix):
    num_sensors = adjacency_matrix.shape[0]
    P_mat = cp.Variable((num_sensors, num_sensors))
    constraints = [
        P_mat >= 0,  # P must be non-negative
        cp.sum(P_mat, axis=1) == 1,  # Each row sums to 1
        P_mat[adjacency_matrix == 0] == 0  # P_ij = 0 if i and j are not connected

    ]
    
    W_bar_terms = []
    for i in range(num_sensors):
        for j in range(num_sensors):
            if adjacency_matrix[i, j] == 1:
                e_i = np.zeros(num_sensors)
                e_j = np.zeros(num_sensors)
                e_i[i] = 1
                e_j[j] = 1
                Wij = np.eye(num_sensors) - 0.5 * np.outer(e_i - e_j, e_i - e_j)
                W_bar_terms.append(P_mat[i, j] * Wij)
                # print("Eigenbasis: ", e_i, e_j)
               
    W_bar = (1 / num_sensors) * cp.sum(W_bar_terms)  # Average Wij matrix
         
    # constraints.append(P_mat  == W_bar)  # P must equal the average Wij matrix
    objective_exp = cp.lambda_max(W_bar - (1/num_sensors)*np.ones((num_sensors, num_sensors)))  # Objective to minimize the largest eigenvalue of W_bar
    
    objective = cp.Minimize(objective_exp)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    P_matrix = P_mat.value
    print(f"Optimal P matrix:\n{P_matrix}")

    ##how many non-negative entries in P_matrix
    num_negative_entries = np.sum(P_matrix < 0)
    print(f"Number of negative entries in P matrix: {num_negative_entries}")

    P_matrix[P_matrix < 0] = 0  # Ensure non-negativity

   
    if P_matrix is None:
        raise ValueError("The optimization problem did not return a valid P matrix.")
    return P_matrix


def randomized_gossip(adjacency_matrix, initial_vals, num_iterations = 50000, tole=1e-6):
    # num_sensors = adjacency_matrix.shape[0]
    # initial_vals = temperature_measurements[:num_sensors][0]
    
    # initial_vals = initial_vals.flatten() 
    initial_vals = np.array(initial_vals)
    current_vals = initial_vals.copy()
    n = len(current_vals)
    estimated_vals = [current_vals.copy()]
    avg_true_vals = np.mean(initial_vals)

    connected_nodes = []
    for i1 in range(n):
        for j1 in range(i1+1, n):
            if adjacency_matrix[i1, j1] == 1:
                connected_nodes.append((i1, j1))
    print(f"Number of connected nodes: {len(connected_nodes)}")
    if connected_nodes:
        print(f"Connected nodes: {connected_nodes[:10]}")
    else:
        print("No connected nodes found in the adjacency matrix.")


    if not connected_nodes:
        print("No connected nodes found in the adjacency matrix.")
        return [], 0

    actual_transmission_count = 0
    tx_counts_natural = [actual_transmission_count]
    for it in range(num_iterations):
        # i, j = connected_nodes[np.random.randint(len(connected_nodes))]  # Randomly select a pair of connected nodes
        
        i = np.random.choice(n)  # Randomly select a node i
        neighbors = np.where(adjacency_matrix[i] == 1)[0]
        if len(neighbors) == 0:
            print(f"Node {i} has no neighbors. Skipping iteration {it}.")
            continue
        j = np.random.choice(neighbors)

        # if adjacency_matrix[i, j] == 1:  # Check if i and j are neighbors
        avg_vals = (current_vals[i] + current_vals[j]) / 2
        current_vals[i] = avg_vals
        current_vals[j] = avg_vals
        estimated_vals.append(current_vals.copy())
        actual_transmission_count += 2
        tx_counts_natural.append(actual_transmission_count)

        if np.linalg.norm(current_vals - avg_true_vals) < tole:
            print(f"Convergence reached at iteration {it}.")
            return estimated_vals, tx_counts_natural  # Return the estimates and the number of iterations
            # break

    # print(f"Maximum iterations reached without convergence.")
    # print("Temp:", temperature_measurements.shape)
    # print(f"Initial values: {initial_vals}")
    # print(f"Final values: {estimated_vals[-1]}")
        # print(f"Estimated average: {np.mean(current_vals)}")
        # print(f"True average: {avg_true_vals}")
        # # print(f"Number of iterations: {num_iterations}")
        # print(f"Actual transmission count: {actual_transmission_count}")


    return estimated_vals, tx_counts_natural

def optimized_randomized_gossip(P_matrix, initial_vals, num_iterations=50000, tole=1e-6):
    n = len(initial_vals)
    current_vals = initial_vals.copy()
    estimated_vals = [current_vals.copy()]
    avg_true_vals = np.mean(initial_vals)

    actual_transmission_count = 0
    txcounts_opt = [actual_transmission_count]
    

    for it in range(num_iterations):
      
        i = np.random.randint(n)  # Randomly select a node i
        neighbors_i = np.where(adjacency_matrix[i] == 1)[0]  # Get neighbors of node i

        if len(neighbors_i) == 0:
            print(f"Node {i} has no neighbors. Skipping iteration {it}.")
            continue  # Skip if node i has no neighbors


        neighbor_probabilities = P_matrix[i, neighbors_i]  # Get transition probabilities for neighbors of node i
        prob_sum = np.sum(neighbor_probabilities)  # Sum of probabilities for node i

        # if prob_sum <= 0:
        #     continue

        norm_prob = neighbor_probabilities /prob_sum
        j = np.random.choice(neighbors_i, p=norm_prob)  # Randomly select a neighbor j based on probabilities

                # if prob_sum > 0:
        avg_vals = (current_vals[i] + current_vals[j]) / 2
        current_vals[i] = avg_vals
        current_vals[j] = avg_vals
        estimated_vals.append(current_vals.copy())
        actual_transmission_count += 2
        txcounts_opt.append(actual_transmission_count)
        # else:
        #     print(f"Warning: No valid transition probabilities for nodes {i} and {j}. Skipping iteration {it}.")
        #     continue
        if np.linalg.norm(current_vals - avg_true_vals) < tole:
            print(f"Convergence reached at iteration {it}.")
            return estimated_vals, txcounts_opt,
        
        # print(f"Maximum iterations reached without convergence.")
        # print(f"Initial values: {initial_vals}")
        # print(f"Final values: {estimated_vals[-1]}")
        # print(f"Estimated average: {np.mean(current_vals)}")
        # print(f"True average: {avg_true_vals}")
        # print(f"Number of iterations: {num_iterations}")
        # print(f"Actual transmission count: {actual_transmission_count}")

    return estimated_vals, txcounts_opt
       

def plot_convergence_random_gossip(estimates_gossip, txcounts_nat, estimates_opt, txcounts_optimized, true_avg):
    # if not estimates:
    #     print("No estimates to plot.")
    #     return
    
    # for i, estimate in enumerate(estimates):
    error_gossip = [np.linalg.norm(estimate - true_avg)**2 for estimate in estimates_gossip]
    error_optimized_gossip = [np.linalg.norm(estimate - true_avg)**2 for estimate in estimates_opt]
    plt.semilogy(txcounts_nat, error_gossip, label='Randomized Gossip', color='blue')
    plt.semilogy(txcounts_optimized, error_optimized_gossip, label='Optimized Randomized Gossip', color='orange')

    # plt.yscale('log')
    # plt.xlabel('Iteration')
    plt.xlabel('Transmission Count')
    plt.ylabel('L2 norm of Error')
    plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')
    plt.title('Convergence Comparison of Randomized Gossip Algorithm')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    

start_time = time.time()
estimated_vals, txcounts = randomized_gossip(adjacency_matrix, sensor_readings)
end_time = time.time()

metrics_random = convergence_summary(sensor_readings, estimated_vals, txcounts)

print(f"[Random Gossip]")
print(f"Iterations to converge: {metrics_random['iterations_to_converge']}")
print(f"Transmissions to converge: {metrics_random['transmissions_to_converge']}")
print(f"Final error: {metrics_random['final_error']:.2e}")

print(f"Time taken for Randomized Gossip: {end_time - start_time:.2f} seconds")
# Calculate the P matrix
start_time = time.time()

P_matrix = calculate_P_matrix(adjacency_matrix)
estimated_vals_P_matrix, txcounts_opt = optimized_randomized_gossip(P_matrix, sensor_readings)

end_time = time.time()
print(f"Time taken for Optimized Randomized Gossip: {end_time - start_time:.2f} seconds")

metrics_opt = convergence_summary(sensor_readings, estimated_vals_P_matrix, txcounts_opt)

print(f"\n[Optimized Random Gossip]")
print(f"Iterations to converge: {metrics_opt['iterations_to_converge']}")
print(f"Transmissions to converge: {metrics_opt['transmissions_to_converge']}")
print(f"Final error: {metrics_opt['final_error']:.2e}")

# Plot the convergence of the Randomized Gossip algorithm

plot_convergence_random_gossip(
    estimated_vals,
    txcounts,
    estimated_vals_P_matrix,
    txcounts_opt,
    true_avg_temp
)




# plot_convergence_random_gossip(true_avg_temp, estimated_vals)
# plot_convergence_random_gossip(true_avg_temp, estimated_vals_P_matrix)


# Run the optimized randomized gossip algorithm
# estimated_vals_P_matrix, num_iterations_P_matrix = optimized_randomized_gossip(P_matrix, sensor_readings)


