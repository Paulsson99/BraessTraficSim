from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from BraessTraficSim.demo import TraficSelfishDrivers
from BraessTraficSim.plot import plot_mean_travel_times


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
    plot_mean_travel_times(ps=[0.01, 0.02, 0.03])
