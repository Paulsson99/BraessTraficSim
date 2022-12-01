from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from large_network import LargeNetwork
from trafficSelfishDrivers import TrafficSelfishDrivers


def plot_mean_travel_times(travel_times: list[list[float]], ax=None, label: str = None) -> None:
    if ax is None:
        _, ax = plt.subplots(1)
    mean_travel_times = [np.mean(travel_time) for travel_time in travel_times]
    ax.plot(np.arange(len(mean_travel_times)), mean_travel_times, label=label)


def generate_road_network(size_of_each_layer):
    """
    Generate a road network in dict{dict} format

        :param size_of_each_layer: List of number of nodes in each layer
        :type size_of_each_layer: list
    """
    large_network = LargeNetwork(size_of_each_layer=size_of_each_layer)
    road_network = large_network.convert_to_graph_to_dict()
    return road_network


def run_simulation(probability_list, road_network):
    """

    Run simulation for each probability in the list

        :param probability_list: A list of probabilities
        :type probability_list: np.ndarray
        :param road_network: Road network in dictionary-dictionary format
        :type road_network: dict{dict}
        :return: traffic as an array

    """
    n_drivers = 100
    n_times_run = 20

    L1 = 20
    L2 = 400

    fig, ax = plt.subplots(1)

    for p in tqdm(probability_list):
        sim = TrafficSelfishDrivers(graph=road_network, N=n_drivers, driver_probability=p)
        pbar = trange(L1 + L2, leave=False)


        traffic_time = []
        for _ in range(n_times_run):
            travel_times, routes = sim.run()
            traffic_time.append(L1)

    plot_mean_travel_times(traffic_time, ax=ax, label=fr"$p={p}$")

    ax.legend()
    plt.show()


def main():
    print()


if __name__ == "__main__":
    probability_list = np.array([0.1])
