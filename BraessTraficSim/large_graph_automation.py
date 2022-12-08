from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from BraessTraficSim.large_network import LargeNetwork
from BraessTraficSim.trafficSelfishDrivers import TrafficSelfishDrivers

# Initialization of the large network
size_of_each_layer = [1, 2, 1]
large_network = LargeNetwork(size_of_each_layer=size_of_each_layer)


def plot_mean_travel_times(travel_times: list[list[float]], ax=None, label: str = None) -> None:
    if ax is None:
        _, ax = plt.subplots(1)
    mean_travel_times = [np.mean(travel_time) for travel_time in travel_times]
    ax.plot(np.arange(len(mean_travel_times)), mean_travel_times, label=label)
    ax.set(xlabel=r'$Time step$', ylabel='Mean travel time')


def generate_initial_road_network(plot_graph: bool, min_max_road_parameters: dict):
    """
    Generate a road network in dict{dict} format by removing some edges.
        :param plot_graph: plot graph
        :param min_max_road_parameters: min max (t0, epsilon, c)
        :return: road network
    """

    large_network.assign_traffic_parameters(min_max_road_parameters=min_max_road_parameters)
    edge_to_remove_list = [(1, 2), (2, 1)]

    for edge_to_remove in edge_to_remove_list:
        large_network.remove_edge(edge_to_remove)

    if plot_graph:
        large_network.plot_initial_graph()
    road_network = large_network.convert_to_graph_to_dict()
    return road_network


def generate_new_road_network(plot_graph: bool, min_max_road_parameters: dict):
    """
    Generate a road network in dict{dict} format by removing some edges.
        :param plot_graph: plot graph
        :param min_max_road_parameters:  Dictionary of min and max (t0, epsilon, c)
        :return: road network
    """
    # large_network.assign_traffic_parameters(min_max_road_parameters=min_max_road_parameters)
    edge_to_add_list = [(1, 2), (2, 1)]

    for edge_to_add in edge_to_add_list:
        large_network.add_edge(edge=edge_to_add,
                               davidson_parameters=generate_random_davidson_parameters(**min_max_road_parameters))


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
        initial_road_network = generate_initial_road_network(plot_graph=plot_initial_graph,)
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
        modified_road_network = generate_new_road_network(plot_graph=plot_initial_graph,
                                                          davidson_parameters=davidson_parameters)
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


def run_until_equilibrium(sim: TrafficSelfishDrivers, max_iteration: int, c: float, transient_time: int, n_averages: int):
    """
    Run the simulation until it find the equilibrium
        :param sim: The trafficSelfish drivers object
        :param max_iteration
        :param c: A parameter that scales the mean when comparing it to the variance
        :param transient_time: The time that we run but ignore
        :param n_averages: Number of times we average the simulation
    """
    travel_time_list = []
    equilibrium_found = True

    for _ in range(transient_time):
        travel_times, _ = sim.run()
        travel_time_list.append(np.mean(travel_times))

    for _ in range(max_iteration):
        if np.var(travel_time_list[-n_averages:]) < c * np.mean(travel_time_list[-n_averages:]):
            break
        travel_times, _ = sim.run()
        travel_time_list.append(np.mean(travel_times))
    else:
        # No equilibrium found
        equilibrium_found = False

    return equilibrium_found, travel_time_list


def generate_random_davidson_parameters(min_t0: float, max_t0: float,
                                        min_eps: float, max_eps: float,
                                        min_c: float, max_c: float):
    davidson_parameters = tuple(np.random.uniform([min_t0, min_eps, min_c], [max_t0, max_eps, max_c]))
    return davidson_parameters


def paradox_prob(n_epochs: int, show_traffic_times: bool):
    #################
    # Parameters of
    # the system
    #################
    n_drivers = 1000
    p = 0.01
    transient_time = 200
    max_iteration = 2000
    n_averages = 30

    #################
    # Parameters of
    # the road network
    #################
    min_max_road_parameters = {
        "min_t0": 10,
        "max_t0": 100,
        "min_eps": 0.1,
        "max_eps": 1,
        "min_c": n_drivers*2,
        "max_c": n_drivers * 10,
    }

    #################
    # Count
    #################
    equilibrium_not_found_count = 0
    paradox_count = 0

    plot_initial_graph = False

    with trange(n_epochs, desc=f"Chance of paradox: {paradox_count / n_epochs:.6f}") as pbar:
        for i in pbar:
            # Initial configuration of the road network
            initial_road_network = generate_initial_road_network(plot_graph=plot_initial_graph,
                                                                 min_max_road_parameters=min_max_road_parameters)
            sim = TrafficSelfishDrivers(road_network=initial_road_network, N=n_drivers, driver_probability=p)
            sim.update_road_network(road_network=initial_road_network)

            # Try to find an equilibrium and add it to the count
            # Run the simulation for the initial configuration of the road network
            equilibrium_found, travel_times1 = run_until_equilibrium(sim=sim, max_iteration=max_iteration,
                                                                     c=0.002, transient_time=transient_time,
                                                                     n_averages=n_averages)
            if show_traffic_times:
                large_network.assign_traffic_to_edges(traffic_in_edges=sim.traffic_count)
                large_network.plot_weighted_graph(driver_probability=p)

            if not equilibrium_found:
                equilibrium_not_found_count += 1
                continue

            # Adding roads to the network and updating it
            modified_road_network = generate_new_road_network(plot_graph=plot_initial_graph,
                                                              min_max_road_parameters=min_max_road_parameters)
            sim.update_road_network(road_network=modified_road_network)

            # Run the simulation for the modified road network
            equilibrium_found, travel_times2 = run_until_equilibrium(sim=sim, max_iteration=max_iteration,
                                                                     c=0.002, transient_time=transient_time,
                                                                     n_averages=n_averages)
            if not equilibrium_found:
                equilibrium_not_found_count += 1
                continue

            first_mean = np.mean(travel_times1[-transient_time // 4:])
            second_mean = np.mean(travel_times2[-transient_time // 4:])

            if first_mean < second_mean:
                paradox_count += 1

            if show_traffic_times:
                print(f'\nParadox count:{paradox_count}')
                print(f"Added road at: {len(travel_times1)}")
                pprint(sim.road_network)

                travel_times = travel_times1 + travel_times2
                plt.subplots()
                plt.plot(range(len(travel_times)), travel_times)
                plt.axvline(x=len(travel_times1), color='black', linestyle='dashed')

                # Plot the traffic after adding a road
                large_network.assign_traffic_to_edges(traffic_in_edges=sim.traffic_count)
                large_network.plot_weighted_graph(driver_probability=p)
                plt.show()

            pbar.set_description(f"Chance of paradox: {paradox_count / (i + 1):.6f}")



def main():
    # plot_initial_graph = True
    # initial_road_network = generate_initial_road_network(plot_graph=plot_initial_graph, davidson_parameters=(1,2,3))
    #modified_road_network = generate_new_road_network(plot_graph=plot_initial_graph, davidson_parameters=(1,2,3))
    #plt.show()

    # run_average_time_simulation(probability_list=p_list)

    paradox_prob(n_epochs=10, show_traffic_times=True)



if __name__ == "__main__":
    main()
