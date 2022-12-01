import numpy as np
from BraessTraficSim.demo import TraficSelfishDrivers, RoadNetwork
from BraessTraficSim.plot import draw_road_network
from tqdm import trange


class Davidsons(TraficSelfishDrivers):

    def __init__(self, graph: RoadNetwork, N: int, c):
        super().__init__(graph, N)
        self.c = c

    def evaluation_edge(self, t0: float, eps: float, u: int):
        return t0 + t0 * eps * u / (self.c - u)


def main():
    c = 600
    eps = 0.5

    road_network = {
        0: {1: (0.01, eps), 2: (45, eps)},
        1: {3: (45, eps), 2: (0.01, eps)},
        2: {3: (0.01, eps), 1: (0.01, eps)},
        3: {}
    }
    drivers = 4000

    trafic = Davidsons(road_network, drivers, c)

    for _ in trange(1000):
        trafic.run()
    travel_times, _ = trafic.run()
    print(f"Average travel time: {np.mean(travel_times)}")

    draw_road_network(road_network, trafic.trafic_count)


if __name__ == '__main__':
    main()
