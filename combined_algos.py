import numpy as np
from preprocessing import *
import matplotlib.pyplot as plt

import cvxpy as cp
def pdmm_algorithm(adjacency_matrix, initial_vals, num_iterations = 50000, c = 1.0, tole = 1e-6):
    
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
                z[(i, j)] = 0
                y[(i, j)] = 0

    transmission_count = 0
    tx_counts = [transmission_count]
    final_iter = num_iterations
    # tx_counts_fail = [0]
    # total_lost = 0
    # # rng = random.Random()

    print(f"Number of edges in the graph: {num_edges}")
    # print(f"Number of connected nodes: {len(connected_nodes)}")

    for it in range(num_iterations):
        # x_prev = x.copy()  # Store the previous values of x
        x_new = np.zeros(n)  # Initialize x_new as a zero vector of size n
        for i in range(n):
            x_new[i] = current_vals[i]  # Update x with the current values
    
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            
            for j in neighbors:
            # Update x based on z
                # A = 1 if i < j else -1
                x_new[i] -= z[(i, j)] 
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
        # tx_counts_fail.append(total_lost)

        if np.linalg.norm(x - avg_true_vals*np.ones(n)) < tole:
        # if np.linalg.norm(x - median_true_vals*np.ones(n)) < tole:
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            return estimated_vals, tx_counts, it # Return the estimates and the number of iterations
        
        if it%100 ==0:
            print(f"Iteration {it}: Estimated average: {np.mean(x)}, True average: {np.mean(avg_true_vals)}, error: {np.linalg.norm(x - avg_true_vals*np.ones(n))}, Transmission count: {transmission_count}")
            # print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")

        
    print("Maximum iterations reached without convergence.")
        
    return estimated_vals, tx_counts, final_iter

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
    # tc = 0
    
    x_new = np.zeros(n)  # Initialize x as a zero vector of size n

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0 and i != j:
                # connected_nodes.append((i, j))
                num_edges += 1
                z[(i, j)] = 0
                y[(i, j)] = 0

    transmission_count = 0
    tx_counts = [transmission_count]
    final_iter = num_iterations
    # txc = [tc]
    print(f"Number of edges in the graph: {num_edges}")
    # print(f"Number of connected nodes: {len(connected_nodes)}")

    for it in range(num_iterations):
        # x_prev = x.copy()  # Store the previous values of x
        x_new = np.zeros(n)  # Initialize x_new as a zero vector of size n

        for i in range(n):
            x_new[i] = current_vals[i]  # Update x with the current values
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            
            # print("Neighbors: ", neighbors)
            # num_neighbors = len(neighbors)
            # print("Neighbors: ", num_neighbors)
            for j in neighbors:
            # Update x based on z
                # A = 1 if i < j else -1
                x_new[i] -=  z[(i, j)] 

            x_new[i] /= (1 + c*len(neighbors))  # Normalize by the number of neighbors
        # Update x with the new values
        x = x_new
        # transmission_count += n  # Count the transmission for each node

        for i in range(n):
            
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                             
                y[(i, j)] = z[(i, j)] + 2*c*x[i]
             
        for i in range(n):
          
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                # A = 1 if i < j else -1
                z[(j, i)] = -y[(i, j)] 

            transmission_count += 1  # Count the transmission

        # Check convergence
        estimated_vals.append(x.copy())
        tx_counts.append(transmission_count)
        # txc.append(tc)
        # print("Transmission count :", transmission_count)
        # print("Transmission count broadcast:", tc)
        if np.linalg.norm(x - avg_true_vals*np.ones(n)) < tole:
        # if np.linalg.norm(x - median_true_vals*np.ones(n)) < tole:
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            return estimated_vals, tx_counts, it # Return the estimates and the number of iterations
        
        if it%100 ==0:
            print(f"Iteration {it}: Estimated average: {np.mean(x)}, True average: {np.mean(avg_true_vals)}, error: {np.linalg.norm(x - avg_true_vals*np.ones(n))}, Transmission count: {transmission_count}")
            # print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")


    print("Maximum iterations reached without convergence.")
        
    return estimated_vals, tx_counts, final_iter

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
    final_iter = num_iterations
    tx_counts_natural = [actual_transmission_count]
    for it in range(num_iterations):
        # i, j = connected_nodes[np.random.randint(len(connected_nodes))]  # Randomly select a pair of connected nodes
        i = np.random.randint(n)  # Randomly select a node i
        neighbors_i = np.where(adjacency_matrix[i] == 1)[0]
        if len(neighbors_i) == 0:
            continue
        j = np.random.choice(neighbors_i)  # Randomly select a neighbor j of node i

        # if adjacency_matrix[i, j] == 1:  # Check if i and j are neighbors
        avg_vals = (current_vals[i] + current_vals[j]) / 2
        current_vals[i] = avg_vals
        current_vals[j] = avg_vals
        estimated_vals.append(current_vals.copy())
        actual_transmission_count += 2
        tx_counts_natural.append(actual_transmission_count)

        if np.linalg.norm(current_vals - avg_true_vals) < tole:
            print(f"Convergence reached at iteration {it}.")
            return estimated_vals, tx_counts_natural, it  # Return the estimates and the number of iterations
            # break

    # print(f"Maximum iterations reached without convergence.")
    # print("Temp:", temperature_measurements.shape)
    # print(f"Initial values: {initial_vals}")
    # print(f"Final values: {estimated_vals[-1]}")
        # print(f"Estimated average: {np.mean(current_vals)}")
        # print(f"True average: {avg_true_vals}")
        # # print(f"Number of iterations: {num_iterations}")
        # print(f"Actual transmission count: {actual_transmission_count}")


    return estimated_vals, tx_counts_natural, final_iter

def optimized_randomized_gossip(adjacency_matrix, P_matrix, initial_vals, num_iterations=50000, tole=1e-6):
    n = len(initial_vals)
    current_vals = np.array(initial_vals, dtype=float).copy()
    estimated_vals = [current_vals.copy()]
    avg_true_vals = np.mean(initial_vals)

    actual_transmission_count = 0
    txcounts_opt = [actual_transmission_count]
    final_iter = num_iterations

    # With the corrected signature, num_iterations is now a proper integer (50000)
    for it in range(num_iterations):
        i = np.random.randint(n)
        neighbors_i = np.where(adjacency_matrix[i] == 1)[0]

        if len(neighbors_i) == 0:
            continue

        # Probabilities of node 'i' choosing a neighbor 'j'
        neighbor_probabilities = P_matrix[i, neighbors_i]
        prob_sum = np.sum(neighbor_probabilities)

        # If prob_sum is 0 or very small, we can't choose a neighbor
        if prob_sum <= 1e-9:
            continue

        # Normalize probabilities for choosing among neighbors
        norm_prob = neighbor_probabilities / prob_sum
        j = np.random.choice(neighbors_i, p=norm_prob)

        avg_vals = (current_vals[i] + current_vals[j]) / 2
        current_vals[i] = avg_vals
        current_vals[j] = avg_vals
        actual_transmission_count += 2
        
        if it % 100 == 0: # Sample estimates to keep the list size manageable
            estimated_vals.append(current_vals.copy())
            txcounts_opt.append(actual_transmission_count)
        
        if np.linalg.norm(current_vals - avg_true_vals) < tole:
            print(f"Optimized Gossip: Convergence reached at iteration {it}.")
            estimated_vals.append(current_vals.copy())
            txcounts_opt.append(actual_transmission_count)
            return estimated_vals, txcounts_opt, it
            
    print("Optimized Gossip: Maximum iterations reached without convergence.")
    
    return estimated_vals, txcounts_opt, final_iter

def async_pdmm_algorithm(adjacency_matrix, initial_vals, num_iterations = 50000, c = 1.0, tole = 1e-6):
    
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
                z[(i, j)] = 0
                y[(i, j)] = 0

    transmission_count = 0
    tx_counts = [transmission_count]
    final_iter = num_iterations
    # tx_counts_fail = [0]
    # total_lost = 0
    # # rng = random.Random()

    print(f"Number of edges in the graph: {num_edges}")
    # print(f"Number of connected nodes: {len(connected_nodes)}")

    for it in range(num_iterations):
        # x_prev = x.copy()  # Store the previous values of x
        x_new = np.zeros(n)  # Initialize x_new as a zero vector of size n
        for i in range(n):
            x_new[i] = current_vals[i]  # Update x with the current values
    
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            
            for j in neighbors:
            # Update x based on z
                # A = 1 if i < j else -1
                x_new[i] -= z[(i, j)] 
            x_new[i] /= (1 + c*len(neighbors))  # Normalize by the number of neighbors
        # Update x with the new values
            x = x_new.copy()

        # for i in range(n):
        #     neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors

            for j in neighbors:
                # A = 1 if i < j else -1
                y[(i, j)] = z[(i , j)] + 2*c*x[i]

        # for i in range(n):
        #     neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors

            for j in neighbors:
                
                # A = 1 if i < j else -1
                z[(j, i)] = -y[(i, j)] 

                transmission_count += 1  # Count the transmission
        # Check convergence
        estimated_vals.append(x.copy())
        tx_counts.append(transmission_count)
        # tx_counts_fail.append(total_lost)

        if np.linalg.norm(x - avg_true_vals*np.ones(n)) < tole:
        # if np.linalg.norm(x - median_true_vals*np.ones(n)) < tole:
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            return estimated_vals, tx_counts, it # Return the estimates and the number of iterations
        
        if it%100 ==0:
            print(f"Iteration {it}: Estimated average: {np.mean(x)}, True average: {np.mean(avg_true_vals)}, error: {np.linalg.norm(x - avg_true_vals*np.ones(n))}, Transmission count: {transmission_count}")
            # print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")

        
    print("Maximum iterations reached without convergence.")
        
    return estimated_vals, tx_counts, final_iter

def async_pdmm_algorithm_broadcast(adjacency_matrix, initial_vals, num_iterations = 50000, c = 1.0, tole = 1e-6):

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
    # tc = 0
    
    x_new = np.zeros(n)  # Initialize x as a zero vector of size n

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0 and i != j:
                # connected_nodes.append((i, j))
                num_edges += 1
                z[(i, j)] = 0
                y[(i, j)] = 0

    transmission_count = 0
    tx_counts = [transmission_count]
    final_iter = num_iterations
    # txc = [tc]
    print(f"Number of edges in the graph: {num_edges}")
    # print(f"Number of connected nodes: {len(connected_nodes)}")

    for it in range(num_iterations):
        # x_prev = x.copy()  # Store the previous values of x
        x_new = np.zeros(n)  # Initialize x_new as a zero vector of size n

        for i in range(n):
            x_new[i] = current_vals[i]  # Update x with the current values
            neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            
            # print("Neighbors: ", neighbors)
            # num_neighbors = len(neighbors)
            # print("Neighbors: ", num_neighbors)
            for j in neighbors:
            # Update x based on z
                # A = 1 if i < j else -1
                x_new[i] -=  z[(i, j)] 

            x_new[i] /= (1 + c*len(neighbors))  # Normalize by the number of neighbors
        # Update x with the new values
            x = x_new
        # transmission_count += n  # Count the transmission for each node

        # for i in range(n):
            
            # neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                             
                y[(i, j)] = z[(i, j)] + 2*c*x[i]
             
        # for i in range(n):
          
            # neighbors = [j for j in range(n) if adjacency_matrix[i, j] > 0 and i != j]  # Get indices of neighbors
            for j in neighbors:
                # A = 1 if i < j else -1
                z[(j, i)] = -y[(i, j)] 

            transmission_count += 1  # Count the transmission

        # Check convergence
        estimated_vals.append(x.copy())
        tx_counts.append(transmission_count)
        # txc.append(tc)
        # print("Transmission count :", transmission_count)
        # print("Transmission count broadcast:", tc)
        if np.linalg.norm(x - avg_true_vals*np.ones(n)) < tole:
        # if np.linalg.norm(x - median_true_vals*np.ones(n)) < tole:
            print(f"Convergence reached at iteration {it}.", f"Transmission count: {transmission_count}")
            return estimated_vals, tx_counts, it # Return the estimates and the number of iterations
        
        if it%100 ==0:
            print(f"Iteration {it}: Estimated average: {np.mean(x)}, True average: {np.mean(avg_true_vals)}, error: {np.linalg.norm(x - avg_true_vals*np.ones(n))}, Transmission count: {transmission_count}")
            # print(f"Iteration {it}: Estimated median: {np.median(x)}, True median: {np.median(median_true_vals)}, error: {np.linalg.norm(x - median_true_vals*np.ones(n))}, Transmission count: {transmission_count}")


    print("Maximum iterations reached without convergence.")
        
    return estimated_vals, tx_counts, final_iter


