import numpy as np
from preprocessing import *
import matplotlib.pyplot as plt
import random
import cvxpy as cp

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

# avg_temperature = np.mean(temperature_measurements[0])
# print(f"Average temperature: {avg_temperature:.2f} °C")

adjacency_matrix = calculate_adjacency_matrix(agent_locations, AGENT_COMM_RANGE)
if not is_graph_connected(adjacency_matrix):
    print("Warning: The graph is not connected.")
else:
    print("The graph is connected.")
visualize_graphs(agent_locations, adjacency_matrix)


# def error_l2norm(x, y):
#     return np.linalg.norm(x - y)**2

def randomized_gossip(adjacency_matrix, initial_vals, num_iterations=50000, tole=1e-6, packet_loss_rate=0):
    initial_vals = np.array(initial_vals)
    current_vals = initial_vals.copy()
    n = len(current_vals)
    errors_steps = []
    transmission_counts_steps = []

    avg_true_vals = np.mean(initial_vals)
    success_prob = dict()  # Dictionary to store success probabilities for each edge
    connected_nodes = []
    for i1 in range(n):
        for j1 in range(i1 + 1, n):
            if adjacency_matrix[i1, j1] == 1:
                connected_nodes.append((i1, j1))
                
                success_prob[(i1, j1)] = 1 - packet_loss_rate
    #print(f"Number of pair of connected nodes (edges): {len(connected_nodes)}")
    #if connected_nodes:
    #    print(f"Connected nodes: {connected_nodes[:10]}")
    #else:
    #    print("No connected nodes found in the adjacency matrix.")
    actual_transmission_count = 0
    total_lost = 0
    errors_steps.append(np.linalg.norm(current_vals - avg_true_vals)**2)
    transmission_counts_steps.append(actual_transmission_count)

    for it in range(num_iterations):

        i, j = connected_nodes[
            np.random.randint(len(connected_nodes))
        ]

        actual_transmission_count += 2

        if np.random.rand() > success_prob[(i, j)]:
            total_lost += 2
            continue  # Simulate packet loss for both nodes

        avg_vals = (current_vals[i] + current_vals[j]) / 2
        current_vals[i] = avg_vals
        current_vals[j] = avg_vals
        # actual_transmission_count += 2  # Count both transmissions
        # errors_steps.append(error_l2norm(current_vals, avg_true_vals))
        errors_steps.append(np.linalg.norm(current_vals - avg_true_vals)**2/ np.linalg.norm(initial_vals)**2)
        transmission_counts_steps.append(actual_transmission_count)
      
        if np.linalg.norm(current_vals - avg_true_vals) < tole:
            print(f"Naive convergence reached at iteration {it}.")
            print(f"Estimated average: {np.mean(current_vals)}")
            print(f"True average: {avg_true_vals}")
            print(f"Actual transmission count: {actual_transmission_count}")
            return errors_steps, transmission_counts_steps, total_lost

    print("Naive Gossip reached max iterations.")
    return errors_steps, transmission_counts_steps, total_lost


def calculate_P_matrix(adjacency_matrix): #from slide 56, Probability matrix that minimizes the second largest eigen val of E(Wbar) to increase speed of convergence
    nb_nodes = adjacency_matrix.shape[0]
    P_mat = cp.Variable((nb_nodes, nb_nodes))

    constraints = [
        P_mat >= 0,                   #Probabilities must be non-negative
        cp.sum(P_mat, axis=1) == 1,   #Sum over j of p_ij = 1
        P_mat[adjacency_matrix == 0] == 0 #p_ij = 0 if (i, j) is not an edge
    ]

    W_bar_terms = []
    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if adjacency_matrix[i, j] == 1:
                e_i = np.zeros(nb_nodes)
                e_j = np.zeros(nb_nodes)
                e_i[i] = 1
                e_j[j] = 1
                Wij = np.eye(nb_nodes) - 0.5 * np.outer(e_i - e_j, e_i - e_j)  #W_ij = I - 0.5 * (e_i - e_j)(e_i - e_j)^T
                W_bar_terms.append(P_mat[i, j] * Wij)


    #W_bar = (1 / nb_nodes) * cp.sum(W_bar_terms) #expectation of Wk
    W_bar = (1/nb_nodes) * sum(W_bar_terms)
    objective_exp = cp.lambda_max(
        W_bar - (1 / nb_nodes) * np.ones((nb_nodes, nb_nodes))   #We can't directly minimize lambda 2 but can make it appear this way.
    )
    objective = cp.Minimize(objective_exp)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    P_matrix = P_mat.value

    #I had some problems with proba <0 even with the constraint, from the solver
    P_matrix[P_matrix < 0] = 0

    return P_matrix

def optimized_randomized_gossip(P_matrix,adjacency_matrix, initial_vals, num_iterations=50000, tole=1e-6, packet_loss_rate=0):
    n = len(initial_vals)
    current_vals = initial_vals.copy()
    errors_steps = []
    transmission_counts_steps = []
    avg_true_vals = np.mean(initial_vals)
    actual_transmission_count = 0
    total_lost = 0
    # success_prob = dict()  # Dictionary to store success probabilities for each edge

    # errors_steps.append(error_l2norm(current_vals, avg_true_vals))
    errors_steps.append(np.linalg.norm(current_vals - avg_true_vals)**2 / np.linalg.norm(initial_vals)**2)
    transmission_counts_steps.append(actual_transmission_count)

    for it in range(num_iterations):
        i=np.random.randint(n) # We select a node randomly

        neighbors_i = np.where(adjacency_matrix[i] == 1)[0] #We find list of neighbors of i

        neighbor_proba = P_matrix[i, neighbors_i] # we find proba for each neighbor

        prob_sum = np.sum(neighbor_proba)
        normalized_proba = neighbor_proba / prob_sum

        j = np.random.choice(neighbors_i, p=normalized_proba)

        actual_transmission_count += 2  # Count both transmissions

        print(np.random.rand(), packet_loss_rate)
        if np.random.rand() < packet_loss_rate:
            total_lost += 2
            continue

        avg_vals = (current_vals[i] + current_vals[j]) / 2
        current_vals[i] = avg_vals
        current_vals[j] = avg_vals

        # actual_transmission_count += 2

        # errors_steps.append(error_l2norm(current_vals, avg_true_vals))
        errors_steps.append(np.linalg.norm(current_vals - avg_true_vals)**2 / np.linalg.norm(initial_vals)**2)
        transmission_counts_steps.append(actual_transmission_count)

        if np.linalg.norm(current_vals - avg_true_vals) < tole:
            print(f"Optimized convergence reached at iteration {it}.")
            print(f"Estimated average: {np.mean(current_vals)}")
            print(f"True average: {avg_true_vals}")
            print(f"Actual transmission count: {actual_transmission_count}")
            return errors_steps, transmission_counts_steps, total_lost

    print("Optimized Randomized Gossip reached max iterations.")
    return errors_steps, transmission_counts_steps, total_lost


def packet_loss_plots(P_matrix, adjacency_matrix, initial_vals, num_iterations = 50000, tole = 1e-6):

    packet_loss_rates = [0.0, 0.25, 0.5, 0.75]
    true_avg_temp = np.mean(initial_vals)
    plt.figure(figsize=(8, 5))

    for loss_rate in packet_loss_rates:
        print(f"\nRunning random gossip with packet loss rate = {loss_rate}")
        estimates, tx_counts, failed = optimized_randomized_gossip(P_matrix,
            adjacency_matrix, initial_vals,
            num_iterations=num_iterations,
            tole=tole,
            packet_loss_rate=loss_rate
        )

        if not estimates:
            print(f"No estimates returned for loss rate = {loss_rate}")
            continue
        
        # if not errors:
        #     print(f"No errors calculated for loss rate = {loss_rate}")
        # errors = [np.linalg.norm(x - true_avg_temp)**2 for x in estimates]
        errors = estimates
        label = f"Loss = {int(loss_rate*100)}%, Failed = {failed}"
        plt.semilogy(tx_counts, errors, label=label)

    plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')
    plt.xlabel('Transmission Count')
    plt.ylabel('L2 Norm of Error')
    plt.title('Random Gossip Convergence with Varying Packet Loss Rates')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the packet loss plots function to visualize the results
# estimated_vals, tx_counts, total_lost = randomized_gossip(
#     adjacency_matrix, sensor_readings,
# )
P_matrix = calculate_P_matrix(adjacency_matrix)

estimated_P_matrix, tx_counts, total_lost = optimized_randomized_gossip(
    P_matrix, adjacency_matrix, sensor_readings)


packet_loss_plots(P_matrix, adjacency_matrix, sensor_readings)


