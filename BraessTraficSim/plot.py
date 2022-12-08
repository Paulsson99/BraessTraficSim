import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('gr',["g", "r"], N=256) 


RoadNetwork = dict[int, dict[int, tuple[float]]]

options = {
    'arrowstyle': '-|>',
    'with_labels': True,
    'arrows': True,
    'width': 4,
    'connectionstyle': 'arc3, rad = 0.1'
}


def draw_road_network(graph: RoadNetwork, traffic: np.ndarray):
    G = nx.MultiDiGraph()
    G.add_nodes_from(graph.keys())

    print(traffic)

    for i, connections in graph.items():
        for j in connections.keys():
            G.add_edge(i, j, weight=traffic[i, j])

    edges, color = zip(*nx.get_edge_attributes(G, 'weight').items())

    fig, ax = plt.subplots(1)
    plot = nx.draw_circular(G, **options, edgelist=edges, edge_color=color, edge_cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=np.max(traffic)))
    sm._A = []
    fig.colorbar(sm, label='Traffic count')
    plt.show()


def plot_mean_travel_times(travel_times: list[list[float]], ax = None, label: str = None) -> None:
    if ax is None:
        _, ax = plt.subplots(1)
    mean_travel_times = [np.mean(travel_time) for travel_time in travel_times]
    ax.plot(np.arange(len(mean_travel_times)), mean_travel_times, label=label)
