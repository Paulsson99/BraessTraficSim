from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from BraessTrafficSim.large_network import LargeNetwork
from BraessTrafficSim.trafficSelfishDrivers import TrafficSelfishDrivers

# Initialization of the large network
size_of_each_layer = [1, 2, 2, 1]
large_network = LargeNetwork(size_of_each_layer=size_of_each_layer)


def plot_mean_travel_times(travel_times: list[list[float]], ax=None, label: str = None) -> None:
    if ax is None:
        _, ax = plt.subplots(1)
    mean_travel_times = [np.mean(travel_time) for travel_time in travel_times]
    ax.plot(np.arange(len(mean_travel_times)), mean_travel_times, label=label)
    ax.set(xlabel=r'$Time step$', ylabel='Mean travel time')


def generate_initial_road_network(plot_graph: bool, davidson_parameters: tuple):
    """
    Generate a road network in dict{dict} format by removing some edges.
        :param plot_graph: plot graph
        :param davidson_parameters:  (t0, epsilon, c)
        :return: road network
    """

    large_network.assign_traffic_parameters(davidson_parameters=davidson_parameters)
    edge_to_remove_list = [(3, 4), (4, 3)]

    for edge_to_remove in edge_to_remove_list:
        large_network.remove_edge(edge_to_remove)

    if plot_graph:
        large_network.plot_initial_graph()
    road_network = large_network.convert_to_graph_to_dict()
    return road_network


def generate_new_road_network(plot_graph: bool, davidson_parameters: tuple):
    """
    Generate a road network in dict{dict} format by removing some edges.
        :param plot_graph: plot graph
        :param davidson_parameters:  (t0, epsilon, c)
        :return: road network
    """
    large_network.assign_traffic_parameters(davidson_parameters=davidson_parameters)
    edge_to_add_list = [(3, 4), (4, 3)]

    for edge_to_add in edge_to_add_list:
        large_network.add_edge(edge_to_add)

    if plot_graph:
        large_network.plot_initial_graph()
        pprint(large_network.convert_to_graph_to_dict())

    road_network = large_network.convert_to_graph_to_dict()
    return road_network


def run_average_time_simulation(probability_list: list):
    """
    Run simulation for each probability in the list and calculate the average time
        :param probability_list: A list of probabilities
        :type probability_list: np.ndarray
    """
    n_drivers = 1000

    # Time steps before and after adding a road
    L1 = 100
    L2 = 400

    plot_initial_graph = False
    davidson_parameters = (1, 2, 3)
    fig, ax = plt.subplots(1)

    for p in tqdm(probability_list):
        initial_road_network = generate_initial_road_network(plot_graph=plot_initial_graph, davidson_parameters=davidson_parameters)
        sim = TrafficSelfishDrivers(road_network=initial_road_network, N=n_drivers, driver_probability=p)
        sim.update_road_network(road_network=initial_road_network)

        pbar = trange(L1 + L2, leave=False)

        traffic_time = []

        for _ in range(L1):
            travel_times, _ = sim.run()
            traffic_time.append(travel_times)
            pbar.update()

        # Plot the traffic before adding the road
        large_network.assign_traffic_to_edges(traffic_in_edges=sim.traffic_count)
        large_network.plot_weighted_graph(driver_probability=p)

        # Adding roads to the network
        modified_road_network = generate_new_road_network(plot_graph=plot_initial_graph,davidson_parameters=davidson_parameters)
        sim.update_road_network(road_network=modified_road_network)

        for _ in range(L2):
            travel_times, _ = sim.run()
            traffic_time.append(travel_times)
            pbar.update()

        # Plot the traffic after adding a road
        large_network.assign_traffic_to_edges(traffic_in_edges=sim.traffic_count)
        large_network.plot_weighted_graph(driver_probability=p)

        plot_mean_travel_times(travel_times=traffic_time, ax=ax, label=fr"$p={p}$")
    ax.legend()
    plt.show()


def main():
    # plot_initial_graph = True
    # initial_road_network = generate_initial_road_network(plot_graph=plot_initial_graph)
    # modified_road_network = generate_new_road_network(plot_graph=plot_initial_graph)
    # plt.show()

    p_list = np.array([0.10])
    run_average_time_simulation(probability_list=p_list)


if __name__ == "__main__":
    main()
