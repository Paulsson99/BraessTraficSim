from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from BraessTraficSim.large_network import LargeNetwork
from BraessTraficSim.trafficSelfishDrivers import TrafficSelfishDrivers

######################################
# Initialization of the large network
######################################

def main(network_size: list[int], nodes_to_connect: list[tuple[int, int]], epochs: int, show_traffic_time: bool):
    large_network = LargeNetwork(size_of_each_layer=network_size)
    paradox_prob(n_epochs=epochs, network=large_network, nodes_to_connect=nodes_to_connect, show_traffic_times=show_traffic_time)


def generate_initial_road_network(min_max_road_parameters: dict, network: LargeNetwork, edges_to_remove_list: list[tuple[int, int]]):
    """
    Generate a road network in dict{dict} format by removing some edges.
        @param min_max_road_parameters: min max (t0, epsilon, c)
        :return: road network
    """

    network.assign_traffic_parameters(min_max_road_parameters=min_max_road_parameters)

    for edge_to_remove in edges_to_remove_list:
        network.remove_edge(edge_to_remove)
        network.remove_edge(tuple(reversed(edge_to_remove)))

    road_network = network.convert_to_graph_to_dict()
    return road_network


def generate_new_road_network(min_max_road_parameters: dict, network: LargeNetwork, edges_to_add_list: list[tuple[int, int]]):
    """
    Generate a road network in dict{dict} format by removing some edges.
        @param min_max_road_parameters:  Dictionary of min and max (t0, epsilon, c)
        :return: road network
    """

    for edge_to_add in edges_to_add_list:
        road_params = generate_random_davidson_parameters(**min_max_road_parameters)
        network.add_edge(edge=edge_to_add, davidson_parameters=road_params)
        network.add_edge(edge=tuple(reversed(edge_to_add)), davidson_parameters=road_params)

    road_network = network.convert_to_graph_to_dict()

    return road_network


def run_until_equilibrium(sim: TrafficSelfishDrivers, max_iteration: int, c: float, transient_time: int,
                          n_averages: int):
    """
    Run the simulation until it find the equilibrium
        @param sim: The trafficSelfish class object
        @param max_iteration after adding the roads
        @param c: A parameter that scales the mean when comparing it to the variance
        @param transient_time: The time that we run first until equilibrium
        @param n_averages: Number of times we average the simulation
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
    """
    Generate random davidson parameters from a dict with specified intervals
    """
    davidson_parameters = tuple(np.random.uniform(low=[min_t0, min_eps, min_c],
                                                  high=[max_t0, max_eps, max_c]))
    return davidson_parameters


def paradox_prob(n_epochs: int, network: LargeNetwork, nodes_to_connect: list[int, int], show_traffic_times: bool):
    """
    Run the simulation and count the occurrence of the paradox  
        @param n_epochs: Number of times we run random generated davidson parameters
        @param show_traffic_times: Plot the graphs after each epoch if true
    """
    #################
    # Parameters of
    # the system
    #################
    n_drivers = 1000
    p = 0.01
    transient_time = 500
    max_iteration = 2000
    n_averages = 30

    #################
    # Parameters of
    # the road network
    #################
    min_max_road_parameters = {
        "min_t0": 10, "max_t0": 100,
        "min_eps": 0.1, "max_eps": 1,
        "min_c": n_drivers * 2, "max_c": n_drivers * 10,
    }

    #################
    # Count
    #################
    equilibrium_not_found_count = 0
    paradox_count = 0

    with trange(n_epochs, desc=f"Chance of paradox: {paradox_count / n_epochs:.6f}") as pbar:
        for i in pbar:
            # Initial configuration of the road network
            initial_road_network = generate_initial_road_network(min_max_road_parameters=min_max_road_parameters, network=network, edges_to_remove_list=nodes_to_connect)
            sim = TrafficSelfishDrivers(road_network=initial_road_network, N=n_drivers, driver_probability=p)
            sim.update_road_network(road_network=initial_road_network)

            # Run the simulation for the initial configuration of the road network
            # Try to find an equilibrium and add it to the count
            equilibrium_found, travel_times1 = run_until_equilibrium(sim=sim, max_iteration=max_iteration,
                                                                     c=0.002, transient_time=transient_time,
                                                                     n_averages=n_averages)

            # Plot the equilibrium road network
            if show_traffic_times:
                network.assign_traffic_to_edges(traffic_in_edges=sim.traffic_count)
                network.plot_weighted_graph()

            # Add to the equilibrium count if found
            if not equilibrium_found:
                equilibrium_not_found_count += 1
                continue

            # Adding roads to the network and updating it
            modified_road_network = generate_new_road_network(min_max_road_parameters=min_max_road_parameters, network=network, edges_to_add_list=nodes_to_connect)
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
                plt.axvline(x=len(travel_times1), color='black', linestyle='dashed',
                            label=f'Roads added at t = {len(travel_times1)}')
                plt.xlabel('Time t'), plt.ylabel('Mean travel time')
                plt.legend()

                # Plot the traffic after adding a road
                network.assign_traffic_to_edges(traffic_in_edges=sim.traffic_count)
                network.plot_weighted_graph()
                plt.show()

            pbar.set_description(f"Chance of paradox: {paradox_count / (i + 1):.6f}")

    print(f"Results for network with size: {network.size_of_each_layer}")
    print(f"Chance of seeing the paradox: {paradox_count / n_epochs:.6f}")
    print(f"No equlibrium found: {equilibrium_not_found_count / n_epochs:.6f}")


if __name__ == "__main__":

    main([1, 2, 1], [(1, 2)], 1, show_traffic_time=True)
