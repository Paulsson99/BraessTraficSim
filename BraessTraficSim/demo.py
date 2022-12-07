import numpy as np
from tqdm import trange


from BraessTraficSim.plot import draw_road_network


RoadNetwork = dict[int, dict[int, tuple[float]]]


class Driver:

    def __init__(self, graph: RoadNetwork, p: float) -> None:
        self.graph = graph
        self.route = self.generate_random_route(0, 3)
        self.p = p
        self.best_travel_time = np.inf
        self.posible_route = None

    def generate_random_route(self, startNode: int, endNode: int):
        route = [startNode]

        while route[-1] != endNode:
            pos_next = self.graph[route[-1]].keys()
            pos_next = [n for n in pos_next if n not in route]
            route.append(np.random.choice(pos_next))
        return route

    def get_route(self):
        if np.random.random() < self.p:
            self.posible_route = self.generate_random_route(0, 3)
            return self.posible_route
        else:
            self.posible_route = None
        return self.route

    def update_route(self, travel_time):
        if self.posible_route is not None:
            if travel_time < self.best_travel_time:
                self.route = self.posible_route
                self.best_travel_time = travel_time
        else:
             self.best_travel_time = travel_time


class TraficSelfishDrivers:

    def __init__(self, graph: RoadNetwork, N: int, driver_prob: float):
        """
        Args:
            graph: A representation of the road network. Keys are node numbers and values are list of connections
            N: Number of cars to simulate
        """
        self.graph = graph
        self.N = N
        self.trafic_count = dict()
        self.drivers = [Driver(graph, driver_prob) for _ in range(N)]

    def evaluation_edge(self, a: float, b: float, u: int):
        """
        Evaluate the time to travel over an edge
        """
        return a * u + b

    def add_road(self, nodeA: int, nodeB: int, params: tuple[float], directed: bool = False) -> None:
        """
        Add a road to the network
        """
        self.graph[nodeA][nodeB] = params
        if not directed:
            self.graph[nodeB][nodeA] = params

    def evaluate_route(self, route):
        """
        Evaluate the total travel time for a route
        """
        total = 0
        for i, j in zip(route[:-1], route[1:]):
            edge = self.graph[i][j]
            edge_trafic = self.trafic_count[i, j]
            total += self.evaluation_edge(*edge, edge_trafic)
            # print(f"Travel time from node {i} to node {j}: {self.evaluation_edge(*edge, edge_trafic)}")

        return total

    def run(self):
        self.trafic_count = np.zeros((len(self.graph), len(self.graph)), dtype=int)
        routes = [driver.get_route() for driver in self.drivers]
        for route in routes:
            i = route[:-1]
            j = route[1:]
            self.trafic_count[i, j] += 1
        travel_times = [self.evaluate_route(route) for route in routes]

        for driver, travel_time in zip(self.drivers, travel_times):
            driver.update_route(travel_time)

        return travel_times, routes


class TraficAntColony:

    def __init__(self, graph: RoadNetwork, N: int):
        """
        Args:
            graph: A representation of the road network. Keys are node numbers and values are list of connections
            N: Number of cars to simulate
        """
        self.graph = graph
        self.N = N
        self.tau = np.ones((len(graph), len(graph)))
        self.alpha = 0.5
        self.trafic_count = dict()
        self.rho = 0.001

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
            edge_trafic = self.trafic_count[i][j]
            total += self.evaluation_edge(*edge, edge_trafic)
            # print(f"Travel time from node {i} to node {j}: {self.evaluation_edge(*edge, edge_trafic)}")

        return total

    def calc_route(self):
        route = [0]
        current_node = 0

        while(current_node != 3):
            next_node = self.next_node(current_node, route)
            if current_node not in self.trafic_count:
                self.trafic_count[current_node] = dict()
            if next_node not in self.trafic_count[current_node]:
                self.trafic_count[current_node][next_node] = 1
            else:
                self.trafic_count[current_node][next_node] += 1

            route.append(next_node)
            current_node = route[-1]

        return route

    def next_node(self, nodeI, route):
        pos_next = self.graph[nodeI].keys()
        # Filter out seen nodes
        unseen_nodes = [n for n in pos_next if n not in route]
        prob = self.tau[nodeI, unseen_nodes] ** self.alpha
        prob /= np.sum(prob)
        return np.random.choice(unseen_nodes, p = prob)

    def run(self):
        self.trafic_count = dict()
        routes = [self.calc_route() for _ in range(self.N)]
        travel_times = [self.evaluate_route(route) for route in routes]
        self.update_tau(routes, travel_times)
        return travel_times, routes

    def update_tau(self, routes: list[list[int]], travel_times: list[float]) -> None:
        delta_tau = np.zeros_like(self.tau)
        for route, travel_time in zip(routes, travel_times):
            for i, j in zip(route[:-1], route[1:]):
                delta_tau[i, j] += 1 / travel_time
        self.tau = (1 - self.rho) * self.tau + delta_tau
        self.tau[self.tau < 0.1] = 0.1


def main():
    road_network = {
        0: {1: (0.01, 0), 2: (0, 45)},
        1: {3: (0, 45), 2: (0, 0)},
        2: {3: (0.01, 0), 1: (0, 0)},
        3: {}
    }
    drivers = 4000

    trafic = TraficSelfishDrivers(road_network, drivers)

    for _ in trange(1000):
        trafic.run()
    travel_times, _ = trafic.run()
    print(f"Average travel time: {np.mean(travel_times)}")

    draw_road_network(road_network, trafic.trafic_count)


if __name__ == '__main__':
    main()
