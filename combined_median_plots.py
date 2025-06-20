import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from combined_median import pdmm_median_algorithm, pdmm_median_algorithm_broadcast, async_pdmm_median_algorithm, async_pdmm_median_algorithm_broadcast

tolerance = 1e-6

def plot_l2_norm_vs_transmission_count(l2_norms, transmission_counts, labels):

    plt.figure(figsize=(10, 6))
    for l2_norm, transmission_count, label in zip(l2_norms, transmission_counts, labels):
        plt.plot(transmission_count, l2_norm, label=label)
    plt.yscale('log')
    plt.xlabel('Transmission Count')
    plt.ylabel('L2 Norm')
    plt.title('L2 Norm vs Transmission Count')
    plt.axhline(y=tolerance, color='red', linestyle='--', label='Tolerance Level (1e-6)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    field_size = 100
    field_range = np.array([[0, field_size], [0, field_size]])
    AGENT_COMM_RANGE = 20
    num_sensors = 149

    print(f"Number of sensors: {num_sensors}")
    print("probability:", 1 - 1 / num_sensors**2)

    agent_locations = np.array([
        0.9 * np.random.uniform(field_range[0, 0], field_range[0, 1], num_sensors),
        0.9 * np.random.uniform(field_range[1, 0], field_range[1, 1], num_sensors)
    ])

    sensor_readings = np.random.uniform(10, 40, num_sensors) + np.random.normal(0, 0.1, num_sensors)
    # true_avg_temp = np.mean(sensor_readings)
    true_median_temp = np.median(sensor_readings)

    adjacency_matrix = calculate_adjacency_matrix(agent_locations, AGENT_COMM_RANGE)

    if not is_graph_connected(adjacency_matrix):
        print("Warning: The graph is not connected.")
    else:
        print("The graph is connected.")
   
    visualize_graphs(agent_locations, adjacency_matrix)

    pdmm_estimates, pdmm_transmissions = pdmm_median_algorithm(adjacency_matrix, sensor_readings, tole = tolerance)
    pdmm_broadcast_estimates, pdmm_broadcast_transmissions = pdmm_median_algorithm_broadcast(adjacency_matrix, sensor_readings, tole = tolerance)
    async_pdmm_estimates, async_pdmm_transmissions = async_pdmm_median_algorithm(adjacency_matrix, sensor_readings, tole = tolerance)
    async_pdmm_broadcast_estimates, async_pdmm_broadcast_transmissions = async_pdmm_median_algorithm_broadcast(adjacency_matrix, sensor_readings, tole = tolerance)

    def compute_l2_norms(estimates, true_val):
        return [np.linalg.norm(x - true_val) for x in estimates]


    estimates_histories = [
        pdmm_estimates,
        pdmm_broadcast_estimates,
        async_pdmm_estimates,
        async_pdmm_broadcast_estimates,
    ]

    l2_norms = [
        compute_l2_norms(pdmm_estimates, true_median_temp),
        compute_l2_norms(pdmm_broadcast_estimates, true_median_temp),
        compute_l2_norms(async_pdmm_estimates, true_median_temp),
        compute_l2_norms(async_pdmm_broadcast_estimates, true_median_temp),
    ]


    transmission_counts = [
        pdmm_transmissions,
        pdmm_broadcast_transmissions,
        async_pdmm_transmissions,
        async_pdmm_broadcast_transmissions,
    ]

    labels = [
        'Synchronous PDMM Algorithm (Unicast)',
        'Synchronous PDMM Algorithm (Broadcast)',
        'Asynchronous PDMM Algorithm (Unicast)',
        'Asynchronous PDMM Broadcast Algorithm (Broadcast)',
    ]

    plot_l2_norm_vs_transmission_count(l2_norms, transmission_counts, labels)

if __name__ == "__main__":
    main()