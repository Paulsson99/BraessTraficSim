import numpy as np
from driver import Driver
road_network = dict[int, dict[int, tuple[float]]]


class TrafficSelfishDrivers:

    def __init__(self, graph: road_network, N: int, driver_probability: float):
        """
        Args:
            graph: A representation of the road network. Keys are node numbers and values are list of connections
            N: Number of cars to simulate
        """
        self.graph = graph
        self.N = N
        self.traffic_count = dict()
        self.drivers = [Driver(graph, driver_probability) for _ in range(N)]

    def evaluation_edge(self, a: float, b: float, u: int):
        """
        Evaluate the time to travel over an edge
        """
        return a * u + b

    def evaluate_route(self, route):
        """
        Evaluate the total travel time for a route
        """
        total = 0
        for i, j in zip(route[:-1], route[1:]):
            edge = self.graph[i][j]
            edge_traffic = self.traffic_count[i, j]
            total += self.evaluation_edge(*edge, edge_traffic)
            # print(f"Travel time from node {i} to node {j}: {self.evaluation_edge(*edge, edge_trafic)}")

        return total

    def run(self):
        self.traffic_count = np.zeros((len(self.graph), len(self.graph)), dtype=int)
        routes = [driver.get_route() for driver in self.drivers]
        for route in routes:
            i = route[:-1]
            j = route[1:]
            self.traffic_count[i, j] += 1
        travel_times = [self.evaluate_route(route) for route in routes]

        for driver, travel_time in zip(self.drivers, travel_times):
            driver.update_route(travel_time)

        return travel_times, routes

