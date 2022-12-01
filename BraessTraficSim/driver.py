import numpy as np

RoadNetwork = dict[int, dict[int, tuple[float]]]


class Driver:

    def __init__(self, graph: RoadNetwork, p: float) -> None:
        self.graph = graph
        self.route = self.generate_random_route(0, 3)
        self.p = p
        self.best_travel_time = np.inf
        self.possible_route = None

    def generate_random_route(self, startNode: int, endNode: int):
        route = [startNode]

        while route[-1] != endNode:
            pos_next = self.graph[route[-1]].keys()
            pos_next = [n for n in pos_next if n not in route]
            route.append(np.random.choice(pos_next))
        return route

    def get_route(self):
        if np.random.random() < self.p:
            self.possible_route = self.generate_random_route(0, 3)
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
