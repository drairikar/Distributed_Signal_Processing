import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from combined_algos import randomized_gossip, optimized_randomized_gossip, pdmm_algorithm, pdmm_algorithm_broadcast, calculate_P_matrix, async_pdmm_algorithm, async_pdmm_algorithm_broadcast

tolerance = 1e-6


def plot_l2_norm_vs_transmission_count(l2_norms, transmission_counts, labels):
    plt.figure(figsize=(10, 6))
    for l2_norm, transmission_count, label in zip(l2_norms, transmission_counts, labels):
        plt.plot(transmission_count, l2_norm, label=label)
    plt.yscale('log')
    plt.xlabel('Transmission Count')
    plt.ylabel('L2 Norm')
    plt.title('L2 Norm vs Transmission Count')
    plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')    
    plt.legend()
    plt.grid(True)
    plt.show()

def l2norm_vs_iterations(l2_norms, iterations, labels):
    plt.figure(figsize=(10, 6))
    for l2_norm, iteration, label in zip(l2_norms, iterations, labels):
        plt.plot(iteration, l2_norm, label=label)

    plt.yscale('log')
    plt.xlabel('Iteration Count')
    plt.ylabel('L2 Norm')
    plt.title('L2 Norm vs Iteration Count')
    plt.axhline(y=1e-6, color='red', linestyle='--', label='Tolerance Level (1e-6)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    field_size = 100
    field_range = np.array([[0, field_size], [0, field_size]])
    AGENT_COMM_RANGE = 20
    num_sensors = 150

    print(f"Number of sensors: {num_sensors}")
    print("probability:", 1 - 1 / num_sensors**2)

    agent_locations = np.array([
        0.9 * np.random.uniform(field_range[0, 0], field_range[0, 1], num_sensors),
        0.9 * np.random.uniform(field_range[1, 0], field_range[1, 1], num_sensors)
    ])

    sensor_readings = np.random.uniform(10, 40, num_sensors) + np.random.normal(0, 0.1, num_sensors)
    true_avg_temp = np.mean(sensor_readings)

    adjacency_matrix = calculate_adjacency_matrix(agent_locations, AGENT_COMM_RANGE)

    if not is_graph_connected(adjacency_matrix):
        print("Warning: The graph is not connected.")
    else:
        print("The graph is connected.")
   
    visualize_graphs(agent_locations, adjacency_matrix)

    # Compute optimal P matrix
    P_matrix = calculate_P_matrix(adjacency_matrix)

    # Run algorithms
    random_estimates, random_transmissions, random_iters = randomized_gossip(adjacency_matrix, sensor_readings, tole = tolerance)
    
    optimized_estimates, optimized_transmissions, opt_iters = optimized_randomized_gossip(adjacency_matrix, P_matrix, sensor_readings, tole = tolerance)
    pdmm_estimates, pdmm_transmissions, pdmm_iters = pdmm_algorithm(adjacency_matrix, sensor_readings, tole = tolerance)
    pdmm_broad_estimates, pdmm_broad_transmissions, pdmm_broad_iters = pdmm_algorithm_broadcast(adjacency_matrix, sensor_readings, tole = tolerance)
    print("Asynch PDMM")
    async_pdmm_estimates, async_pdmm_transmissions, async_pdmm_iters = async_pdmm_algorithm(adjacency_matrix, sensor_readings, tole = tolerance)
    print("Asynch PDMM Broadcast")
    async_pdmm_broad_estimates, async_pdmm_broad_transmissions, async_broad_iters = async_pdmm_algorithm_broadcast(adjacency_matrix, sensor_readings, tole = tolerance)

    # Compute L2 norms
    def compute_l2_norms(estimates, true_val):
        return [np.linalg.norm(x - true_val) for x in estimates]
    
    rec_estimates =[
        random_estimates,
        optimized_estimates,
        pdmm_estimates,
        pdmm_broad_estimates,
        async_pdmm_estimates,
        async_pdmm_broad_estimates,
    ]

    l2_norms = [
        compute_l2_norms(random_estimates, true_avg_temp),
        compute_l2_norms(optimized_estimates, true_avg_temp),
        compute_l2_norms(pdmm_estimates, true_avg_temp),
        compute_l2_norms(pdmm_broad_estimates, true_avg_temp),
        compute_l2_norms(async_pdmm_estimates, true_avg_temp),
        compute_l2_norms(async_pdmm_broad_estimates, true_avg_temp),
    ]

    transmission_counts = [
        random_transmissions,
        optimized_transmissions,
        pdmm_transmissions,
        pdmm_broad_transmissions,
        async_pdmm_transmissions,
        async_pdmm_broad_transmissions,
    ]

    labels = [
        'Random Gossip',
        'Optimized Gossip',
        'PDMM (Synchronous, Unicast)',
        'PDMM (Synchronous, Broadcast)',
        'Asynchronous PDMM (Unicast)',
        'Asynchronous PDMM (Broadcast)'
    ]

    iteration_counts = [
        random_iters,
        opt_iters,
        pdmm_iters,
        pdmm_broad_iters,
        async_pdmm_iters,
        async_broad_iters
    ]

    for label, iters, trans in zip(labels, iteration_counts, transmission_counts):
        print(f"{label:<25}: Converged in {iters} iterations with {trans[-1]} transmissions.")

    plot_l2_norm_vs_transmission_count(l2_norms, transmission_counts, labels)
    # Plot L2 norm vs iterations
    l2norm_vs_iterations(l2_norms, rec_estimates, labels)
    

if __name__ == "__main__":
    main()
