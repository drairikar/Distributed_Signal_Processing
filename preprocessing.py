import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D


def calculate_adjacency_matrix(agent_locations, communication_range):
    # agent_locations: [GP_INPUT_DIM, NUM_AGENTS]
    # communication_range: float

    num_agents = agent_locations.shape[1]
    adjacency_matrix = np.zeros((num_agents, num_agents), dtype=int)
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            distance = np.linalg.norm(agent_locations[:, i] - agent_locations[:, j])
            if distance <= communication_range:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix

def is_graph_connected(adjacency_matrix):

    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Compute eigenvalues of the Laplacian matrix
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix)

    # Count zero eigenvalues
    zero_eigenvalues = np.sum(np.isclose(eigenvalues, 0))

    # Graph is connected if there is exactly one zero eigenvalue
    return zero_eigenvalues == 1

def number_of_sensors(communication_range, field_size):
    
    num_sensors = 2
    
    while communication_range**2 < 2*np.log(num_sensors)/num_sensors*field_size**2:
        num_sensors += 1
    return num_sensors

def temperature_generator(field_size = 100, field_resolution = (100, 100), min_temp = 10, max_temp = 40):
    """
    Generates a temperature field as a 2D numpy array
    - field_size: size of the field (assumed square)
    - field_resolution: resolution of the field (number of points along each axis)
    """
    x = np.linspace(0, field_size, field_resolution[0])
    y = np.linspace(0, field_size, field_resolution[1])
    noise = np.random.normal(0, 0.1, size=(field_resolution[0], field_resolution[1]))
    # Create a meshgrid for the field

    mesh_x, mesh_y = np.meshgrid(x, y, indexing='ij')
    
    # Generate a random temperature field
    temperature_field = np.random.uniform(min_temp, max_temp, size=mesh_x.shape) + noise
    
    return temperature_field, mesh_x, mesh_y


def visualize_graphs(agent_locations, adjacency_matrix):
    num_agents = agent_locations.shape[1]
    colors = plt.get_cmap("tab10", num_agents)  # Use a colormap with N unique colors

    # field_size = temperature_generator.shape[0]
    field_size = 100
    field_resolution = (field_size, field_size)
    field_range = np.array([[0, field_size], [0, field_size]])
    x1 = np.linspace(field_range[0,0], field_range[0,1], field_resolution[0])
    x2 = np.linspace(field_range[1,0], field_range[1,1], field_resolution[1])

    plt.figure(figsize=(8, 6))
    # Create a meshgrid for the field
    mesh_x1, mesh_x2 = np.meshgrid(x1, x2, indexing='ij')

    actual_temperature_data, _, _ = temperature_generator(
        field_size=field_size,
        field_resolution=field_resolution
    )
    

    temp_plot = plt.contourf(mesh_x1, mesh_x2, actual_temperature_data, levels=50, cmap='plasma', alpha=0.7)
    plt.colorbar(temp_plot, label='Temperature (°C)')
    # Overlay agents and sample locations and their interpolated values

    # Plot the agents
    # plt.figure(figsize=(3, 3))
    for i in range(num_agents):
        plt.scatter(agent_locations[0, i], agent_locations[1, i], marker = 'o', s = 70, color = colors(i),edgecolors='black', zorder = 3)
        # plt.scatter(sample_locations_list[i][0,:], sample_locations_list[i][1,:], marker = 'o', color=colors(i))

    # Plot edges
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if adjacency_matrix[i, j] == 1:
                plt.plot([agent_locations[0, i], agent_locations[0, j]],
                         [agent_locations[1, i], agent_locations[1, j]],
                         'gray', linewidth=1, zorder=1)
    
    plt.axis("equal")
    plt.title("Sensor network over the temperature field")
    if os.environ.get("SAVE_FIGS") == "1":
        os.makedirs("figures", exist_ok=True)
        plt.savefig(os.path.join("figures", "network.png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()