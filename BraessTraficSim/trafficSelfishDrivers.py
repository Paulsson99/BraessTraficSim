import numpy as np
from driver import Driver
from pprint import pprint

road_network_structure = dict[int, dict[int, tuple[float]]]


class TrafficSelfishDrivers:

    def __init__(self, road_network: road_network_structure, N: int, driver_probability: float):
        """
        Args:
            road_network: A representation of the road network. Keys are node numbers and values are list of connections
            N: Number of cars to simulate
        """
        self.road_network = road_network
        self.N = N
        self.traffic_count = dict()
        self.drivers = [Driver(road_network, driver_probability) for _ in range(N)]
        self.driver_probability = driver_probability

    def evaluation_edge(self, t0: float, eps: float, c: float, u: int):
        """
        Compute travel time depending on the traffic, i.e. the parameters
        """
        t_max = self.N * t0
        if u >= c:
            return t_max
        return min(t_max, t0 + t0 * eps * u / (c - u))

    def evaluate_route(self, route):
        """
        Evaluate the total travel time for a route
        """
        total = 0
        for i, j in zip(route[:-1], route[1:]):
            edge = self.road_network[i][j]
            edge_traffic = self.traffic_count[i, j]
            total += self.evaluation_edge(*edge, edge_traffic)
            # print(f"Travel time from node {i} to node {j}: {self.evaluation_edge(*edge, edge_trafic)}")

        return total

    def update_road_network(self, road_network):
        """
        Update the road network for each driver to a new one
        """
        self.road_network = road_network
        for driver in self.drivers:
            driver.update_road_network(road_network=road_network)

    def run(self):
        self.traffic_count = np.zeros((len(self.road_network), len(self.road_network)), dtype=int)
        routes = [driver.get_route() for driver in self.drivers]
        for route in routes:
            i = route[:-1]
            j = route[1:]
            self.traffic_count[i, j] += 1
        travel_times = [self.evaluate_route(route) for route in routes]

        for driver, travel_time in zip(self.drivers, travel_times):
            driver.update_route(travel_time)

        return travel_times, routes
