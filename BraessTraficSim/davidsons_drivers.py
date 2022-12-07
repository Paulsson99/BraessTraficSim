import matplotlib.pyplot as plt
import numpy as np
from BraessTraficSim.demo import TraficSelfishDrivers, RoadNetwork
from BraessTraficSim.plot import draw_road_network
from tqdm import trange, tqdm


class Davidsons(TraficSelfishDrivers):

    def __init__(self, graph: RoadNetwork, N: int, driver_prob: float):
        super().__init__(graph, N, driver_prob)

    def evaluation_edge(self, t0: float, eps: float, c: float, u: int):
        t_max = 1000 * t0
        if u >= c:
            return t_max
        return min(t_max, t0 + t0 * eps * u / (c - u))


def main():
    drivers = 4000
    transient_time = 100
    simulation_time = 1000
    p = 0.01
    simulation_on = False

    c_list = np.array(range(1000, 4000, 500))
    eps_list = np.arange(0.2, 0.8, 0.1)
    time_difference = np.zeros((len(c_list), len(eps_list)), dtype=np.float_)

    c_std = 2500
    eps_std = 0.5
    i = 0
    if simulation_on:
        for c in tqdm(c_list):
            for j, eps in enumerate(eps_list):
                road_network = {
                    0: {1: (0.01, eps_std, c_std), 2: (0.01, eps_std, c_std)},
                    1: {3: (0.01, eps_std, c_std)},
                    2: {3: (0.01, eps_std, c_std)},
                    3: {}
                }

                trafic = Davidsons(road_network, drivers, driver_prob=p)
                trafic.run()

                for _ in range(transient_time):
                    trafic.run()
                travel_times0, _ = trafic.run()

                # Add the two new roads between node 1 and 2
                trafic.add_road(1, 2, (0.01, eps, c))

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
    c_label = ['{:.0f}'.format(c) for c in c_list]
    eps_label = ['{:.2f}'.format(eps) for eps in eps_list]
    ax.set_xticks(range(len(c_list)), c_label)
    ax.set_yticks(range(len(eps_list)), eps_label)
    ax.autoscale(tight=True)
    ax.set_xlabel('Capacity $c$')
    ax.set_ylabel(r'$\epsilon$')
    fig.colorbar(pos, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
