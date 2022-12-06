import matplotlib.pyplot as plt
import numpy as np
from BraessTraficSim.demo import TraficSelfishDrivers, RoadNetwork
from BraessTraficSim.plot import draw_road_network
from tqdm import trange, tqdm


class Davidsons(TraficSelfishDrivers):

    def __init__(self, graph: RoadNetwork, N: int):
        super().__init__(graph, N)

    def evaluation_edge(self, t0: float, eps: float, c: float, u: int):
        t_max = 1000 * t0
        if u >= c:
            return t_max
        return min(t_max, t0 + t0 * eps * u / (c - u))


def main():
    drivers = 4000
    simulation_time = 1000
    simulation_on = False

    c_list = np.array(range(1000, 2000, 500)) #1000, 4000, 500
    eps_list = np.arange(0.2, 0.4, 0.1) #0.2, 0.8, 0.1
    time_difference = np.zeros((len(c_list), len(eps_list)), dtype=np.float_)

    c_std = 2500
    eps_std = 0.5
    i = 0
    if simulation_on:
        for c in tqdm(c_list):
            for j, eps in enumerate(eps_list):
                road_network = {
                    0: {1: (0.01, eps_std, c_std), 2: (0.01, eps_std, c_std)},
                    1: {3: (0.01, eps_std, c_std), 2: (0.01, eps, c)},
                    2: {3: (0.01, eps_std, c_std), 1: (0.01, eps, c)},
                    3: {}
                }

                trafic = Davidsons(road_network, drivers)
                trafic.run()
                travel_times0, _ = trafic.run()
                for _ in trange(simulation_time):
                    trafic.run()
                travel_times, _ = trafic.run()
                time_difference[i, j] = np.mean(travel_times) - np.mean(travel_times0)
                # print(f"Average travel time: {np.mean(travel_times)}")
            i += 1

        np.save('time_difference.npy', time_difference)

    time_difference = np.load('time_difference.npy')
    # draw_road_network(road_network, trafic.trafic_count)
    fig, ax = plt.subplots(layout='constrained')
    pos = ax.imshow(time_difference, origin='lower')
    ax.set_xticks(range(len(c_list)), c_list)
    ax.set_yticks(range(len(eps_list)), eps_list)
    fig.colorbar(pos, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
