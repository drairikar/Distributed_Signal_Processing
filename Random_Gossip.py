from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

field_size = 100
# field_resolution = (field_size, field_size)
field_range = np.array([[0, field_size], [0, field_size]])
# num_sensors = 10
AGENT_COMM_RANGE = 30
            
num_sensors = number_of_sensors(AGENT_COMM_RANGE, field_size)
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
   
    if P_matrix is None:
        raise ValueError("The optimization problem did not return a valid P matrix.")
    return P_matrix


def randomized_gossip(adjacency_matrix, initial_vals, num_iterations = 50000, tole=1e-4):
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
    for it in range(num_iterations):
        i, j = connected_nodes[np.random.randint(len(connected_nodes))]  # Randomly select a pair of connected nodes

        # if adjacency_matrix[i, j] == 1:  # Check if i and j are neighbors
        avg_vals = (current_vals[i] + current_vals[j]) / 2
        current_vals[i] = avg_vals
        current_vals[j] = avg_vals
        estimated_vals.append(current_vals.copy())
        actual_transmission_count += 1

        if np.linalg.norm(current_vals - avg_true_vals) < tole:
            print(f"Convergence reached at iteration {it}.")
            return estimated_vals, actual_transmission_count  # Return the estimates and the number of iterations
            # break

    # print(f"Maximum iterations reached without convergence.")
    # print("Temp:", temperature_measurements.shape)
    # print(f"Initial values: {initial_vals}")
    # print(f"Final values: {estimated_vals[-1]}")
        print(f"Estimated average: {np.mean(current_vals)}")
        print(f"True average: {avg_true_vals}")
        # print(f"Number of iterations: {num_iterations}")
        print(f"Actual transmission count: {actual_transmission_count}")


    return estimated_vals, actual_transmission_count

def optimized_randomized_gossip(P_matrix, initial_vals, num_iterations=50000, tole=1e-6):
    n = len(initial_vals)
    current_vals = initial_vals.copy()
    estimated_vals = [current_vals.copy()]
    avg_true_vals = np.mean(initial_vals)

    actual_transmission_count = 0

    for it in range(num_iterations):
        i, j = np.random.choice(n, size=2, replace=False)  # Randomly select two nodes
        probability = P_matrix[i, j]  # Get the transition probability from P_matrix
        prob_sum = np.sum(probability)  # Sum of probabilities for node i
        if prob_sum > 0:
            avg_vals = (current_vals[i] + current_vals[j]) / 2
            current_vals[i] = avg_vals
            current_vals[j] = avg_vals
            estimated_vals.append(current_vals.copy())
            actual_transmission_count += 1
        else:
            print(f"Warning: No valid transition probabilities for nodes {i} and {j}. Skipping iteration {it}.")
            continue
        if np.linalg.norm(current_vals - avg_true_vals) < tole:
            print(f"Convergence reached at iteration {it}.")
            return estimated_vals, it + 1
        
        # print(f"Maximum iterations reached without convergence.")
        # print(f"Initial values: {initial_vals}")
        # print(f"Final values: {estimated_vals[-1]}")
        print(f"Estimated average: {np.mean(current_vals)}")
        print(f"True average: {avg_true_vals}")
        print(f"Number of iterations: {num_iterations}")
        print(f"Actual transmission count: {actual_transmission_count}")

    return estimated_vals, actual_transmission_count
       

def plot_convergence_random_gossip(true_avg, estimates):
    if not estimates:
        print("No estimates to plot.")
        return
    
    # for i, estimate in enumerate(estimates):
    for i , estimate in estimates.items():
        errors = [np.linalg.norm(np.array(x_vals) - true_avg)**2 for x_vals in estimate]
        plt.plot(errors, label=i)
    plt.yscale('log')
    # plt.xlabel('Iteration')
    plt.xlabel('Transmission Count')
    plt.ylabel('L2 norm of Error')
    plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')
    plt.title('Convergence Comparison of Randomized Gossip Algorithm')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    

# Run the randomized gossip algorithm
# estimated_vals, num_iterations = randomized_gossip(adjacency_matrix, temperature_measurements[:num_sensors])
# estimated_vals, num_iterations = randomized_gossip(adjacency_matrix, sensor_readings)
estimated_vals, transmission_count = randomized_gossip(adjacency_matrix, sensor_readings)
# Calculate the P matrix
P_matrix = calculate_P_matrix(adjacency_matrix)
# Run the optimized randomized gossip algorithm
# estimated_vals_P_matrix, num_iterations_P_matrix = optimized_randomized_gossip(P_matrix, sensor_readings)
estimated_vals_P_matrix, transmission_count_P_matrix = optimized_randomized_gossip(P_matrix, sensor_readings)

plot_convergence_random_gossip(
    true_avg_temp, 
    {
        'Randomized Gossip': estimated_vals,
        'Optimized Randomized Gossip': estimated_vals_P_matrix
    }
)
# plot_convergence_random_gossip(true_avg_temp, estimated_vals)
# plot_convergence_random_gossip(true_avg_temp, estimated_vals_P_matrix)





