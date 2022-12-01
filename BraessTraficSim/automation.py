from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from large_network import LargeNetwork
from trafficSelfishDrivers import TrafficSelfishDrivers

size_of_each_layer = [1, 3, 3, 1]
large_network = LargeNetwork(size_of_each_layer=size_of_each_layer)


def plot_mean_travel_times(travel_times: list[list[float]], ax=None, label: str = None) -> None:
    if ax is None:
        _, ax = plt.subplots(1)
    mean_travel_times = [np.mean(travel_time) for travel_time in travel_times]
    ax.plot(np.arange(len(mean_travel_times)), mean_travel_times, label=label)


def generate_initial_road_network():
    """
    Generate a road network in dict{dict} format by removing some edges.
    """
    pprint(large_network.convert_to_graph_to_dict())
    edge_to_remove_list = [(1, 4), (3, 6)]
    for edge_to_remove in edge_to_remove_list:
        large_network.remove_edge(edge_to_remove)

    #  large_network.plot_initial_graph()
    road_network = large_network.convert_to_graph_to_dict()
    return road_network


def generate_new_road_network():
    """
    Generate a road network in dict{dict} format by adding some edges.
    """
    edge_to_add_list = [(1, 4)]
    for edge_to_add in edge_to_add_list:
        large_network.add_edge(edge_to_add)

    #  large_network.plot_initial_graph()
    road_network = large_network.convert_to_graph_to_dict()
    return road_network


def run_simulation(probability_list):
    """

    Run simulation for each probability in the list and calculate the average time

        :param probability_list: A list of probabilities
        :type probability_list: np.ndarray
        :return: traffic as an array

    """
    n_drivers = 10

    L1 = 20
    L2 = 400

    initial_road_network = generate_initial_road_network()
    modified_road_network = generate_new_road_network()

    fig, ax = plt.subplots(1)

    for p in tqdm(probability_list):
        sim = TrafficSelfishDrivers(road_network=initial_road_network, N=n_drivers, driver_probability=p)
        pbar = trange(L1 + L2, leave=False)

        traffic_time = []

        for _ in range(L1):
            travel_times, _ = sim.run()
            traffic_time.append(travel_times)
            pbar.update()

        sim.update_road_network(graph=modified_road_network)
        for _ in range(L2):
            travel_times, _ = sim.run()
            traffic_time.append(travel_times)
            pbar.update()

    plot_mean_travel_times(travel_times=traffic_time, ax=ax, label=fr"$p={p}$")

    ax.legend()
    plt.show()


def main():
    p_list = np.array([0.1])
    run_simulation(probability_list=p_list)


if __name__ == "__main__":
    main()
