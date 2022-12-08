import numpy as np

road_network_structure = dict[int, dict[int, tuple[float]]]


class Driver:

    def __init__(self, road_network: road_network_structure, p: float) -> None:
        self.road_network = road_network
        self.route = self.generate_random_route(start_node=min(road_network.keys()), end_node=max(road_network.keys()))
        self.p = p
        self.best_travel_time = np.inf
        self.possible_route = None

    def generate_random_route(self, start_node: int, end_node: int):
        route = [start_node]

        while route[-1] != end_node:
            pos_next = self.road_network[route[-1]].keys()
            pos_next = [n for n in pos_next if n not in route]
            route.append(np.random.choice(pos_next))
        return route

    def update_road_network(self, road_network):
        """
        Update the road network to a new one
        """
        self.road_network = road_network

    def get_route(self):
        if np.random.random() < self.p:
            self.possible_route = self.generate_random_route(start_node=min(self.road_network.keys()), end_node=max(self.road_network.keys()))
            return self.possible_route
        else:
            self.possible_route = None
        return self.route

    def update_route(self, travel_time):
        if self.possible_route is not None:
            if travel_time < self.best_travel_time:
                self.route = self.possible_route
                self.best_travel_time = travel_time
        else:
            self.best_travel_time = travel_time
