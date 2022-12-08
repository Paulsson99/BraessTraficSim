import matplotlib.pyplot as plt
import numpy as np
from BraessTraficSim.demo import TraficSelfishDrivers, RoadNetwork
from BraessTraficSim.plot import draw_road_network
from tqdm import trange, tqdm


class Davidsons(TraficSelfishDrivers):

    def __init__(self, graph: RoadNetwork, N: int, driver_prob: float):
        super().__init__(graph, N, driver_prob)

    def evaluation_edge(self, t0: float, eps: float, c: float, u: int):
        return t0 + t0 * eps * u / (c - u)


def run_until_equlibrium(trafic: Davidsons, max_iters: int, c: float, transient: int, average: int) -> list[float]:
    travel_times = []
    equlibrium_found = True
    for _ in range(transient):
        travel_t, _ = trafic.run()
        travel_times.append(np.mean(travel_t))

    for _ in range(max_iters):
        if np.var(travel_times[-average:]) < c * np.mean(travel_times[-average:]):
            break
        travel_t, _ = trafic.run()
        travel_times.append(np.mean(travel_t))
    else:
        # No equlibrium found
        equlibrium_found = False

    return equlibrium_found, travel_times


def random_road(min_eps: float, max_eps: float, min_c: float, max_c: float, min_t0: float, max_t0: float) -> tuple[float, float, float]:
    return tuple(np.random.uniform([min_t0, min_eps, min_c], [max_t0, max_eps, max_c]))


def paradox_prob(epochs: int, show_traffic_times: bool = False):
    drivers = 1000

    road_params = {
        "min_eps": 0.1,
        "max_eps": 1,
        "min_c": drivers*2,
        "max_c": drivers * 10,
        "min_t0": 10,
        "max_t0": 100
    }

    transient_time = 500
    averages = 30
    max_iter = 2000
    p = 0.01

    equlibrium_not_found = 0
    paradox_count = 0
    
    with trange(epochs, desc=f"Chance of paradox: {paradox_count / epochs:.6f}") as pbar:
        for i in pbar:
            road_network = {
                0: {1: random_road(**road_params), 2: random_road(**road_params)},
                1: {3: random_road(**road_params)},
                2: {3: random_road(**road_params)},
                3: {}
            }

            trafic = Davidsons(road_network, drivers, driver_prob=p)

            equlibrium_found, travel_times1 = run_until_equlibrium(trafic, max_iters=max_iter, c=0.002, transient=transient_time, average=averages)
            if not equlibrium_found:
                equlibrium_not_found += 1
                continue
            trafic.add_road(nodeA=1, nodeB=2, params=random_road(**road_params), directed=True)
            equlibrium_found, travel_times2 = run_until_equlibrium(trafic, max_iters=max_iter, c=0.002, transient=transient_time, average=averages)
            if not equlibrium_found:
                equlibrium_not_found += 1
                continue

            first_mean = np.mean(travel_times1[-transient_time//4:])
            second_mean = np.mean(travel_times2[-transient_time//4:])

            if first_mean < second_mean:
                paradox_count += 1
            
            if show_traffic_times:
                print(paradox_count)
                print(trafic.graph)
                print(f"Added road at: {len(travel_times1)}")
                travel_times = travel_times1 + travel_times2
                plt.plot(range(len(travel_times)), travel_times)
                plt.show()
    
            pbar.set_description(f"Chance of paradox: {paradox_count / (i + 1):.6f}")
    
    print(f"Chance of seeing the paradox: {paradox_count / epochs:.6f}")
    print(f"No equlibrium found: {equlibrium_not_found / epochs:.6f}")


def heat_map():
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
    paradox_prob(epochs=1000, show_traffic_times=False)
