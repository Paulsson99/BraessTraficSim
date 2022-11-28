from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

from BraessTraficSim.demo import TraficSelfishDrivers
from BraessTraficSim.plot import plot_mean_travel_times


def plot_break_even(ps: np.ndarray, averages: int):
    fig, ax = plt.subplots(1)

    L1 = 20
    L2 = 10000

    Ts = []

    outer_pbar = trange(len(ps) * averages)

    for p in ps:
        T_tot = 0
        for _ in range(averages):
            road_network = {
                0: {1: (0.01, 0), 2: (0, 45)},
                1: {3: (0, 45)},
                2: {3: (0.01, 0)},
                3: {}
            }
            drivers = 4000
            sim = TraficSelfishDrivers(graph=road_network, N=drivers, driver_prob=p)

            inner_pbar = trange(L1 + L2, leave=False)

            traffic_time_1 = []
            for _ in range(L1):
                travel_times, _ = sim.run()
                traffic_time_1.append(travel_times)
                inner_pbar.update()
            sim.add_road(nodeA=1, nodeB=2, params=(0, 0))
            mean1 = np.mean([np.mean(travel_time) for travel_time in traffic_time_1])

            for i in range(L2):
                travel_times, _ = sim.run()
                mean_travel_time = np.mean(travel_times)
                inner_pbar.update()
                if mean_travel_time > mean1 * 1.03:
                    break

            inner_pbar.close()
            T_tot += i
            
            outer_pbar.update()

        Ts.append(T_tot / averages)
    
    outer_pbar.close()

    # Save the data
    np.savetxt('ps.csv', ps, delimiter=',')
    np.savetxt('Ts.csv', Ts, delimiter=',')

    ax.plot(ps, Ts)
    ax.set_xscale('log')
    plt.show()

def plot_travel_times(ps: list[float]):
    fig, ax = plt.subplots(1)
    for p in tqdm(ps):
        road_network = {
            0: {1: (0.01, 0), 2: (0, 45)},
            1: {3: (0, 45)},
            2: {3: (0.01, 0)},
            3: {}
        }
        drivers = 4000
        sim = TraficSelfishDrivers(graph=road_network, N=drivers, driver_prob=p)

        L1 = 20
        L2 = 400

        pbar = trange(L1 + L2, leave=False)

        traffic_time = []
        for _ in range(L1):
            travel_times, _ = sim.run()
            traffic_time.append(travel_times)
            pbar.update()
        sim.add_road(nodeA=1, nodeB=2, params=(0, 0))
        for _ in range(L2):
            travel_times, _ = sim.run()
            traffic_time.append(travel_times)
            pbar.update()
    
        plot_mean_travel_times(traffic_time, ax=ax, label=fr"$p={p}$")

    ax.legend()
    plt.show()


if __name__ == '__main__':
    # plot_mean_travel_times(ps=[0.01, 0.02, 0.03])
    plot_break_even(np.logspace(-3, 0, 50), averages=3)
