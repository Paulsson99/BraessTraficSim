import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mycolorpy import colorlist as mcp
from pprint import pprint
from matplotlib.colors import LinearSegmentedColormap

from trafficSelfishDrivers import TrafficSelfishDrivers


class LargeNetwork:
    def __init__(self, size_of_each_layer: list, traffic: np.ndarray):
        """
        Create a large network
        Args:
            size_of_each_layer: A list of how many nodes per layer
            traffic: Traffic matrix between the edges
        """
        self.G = None
        self.number_of_layers = len(size_of_each_layer)
        self.size_of_each_layer = size_of_each_layer
        self.layer_colors = mcp.gen_color(cmap="winter", n=self.number_of_layers)
        self.generate_multilayered_graph(*self.size_of_each_layer)
        self.traffic = traffic

    def generate_multilayered_graph(self, *size_of_each_layer):
        """
        Generate a multilayered graph
        Args:
            size_of_each_layer: A list of how many nodes per layer
        """
        extents = nx.utils.pairwise(itertools.accumulate((0,) + size_of_each_layer))
        layers = [range(start, end) for start, end in extents]
        self.G = nx.Graph()

        for (i, layer) in enumerate(layers):
            self.G.add_nodes_from(layer, layer=i)

        for layer1, layer2 in nx.utils.pairwise(layers):
            self.G.add_edges_from(itertools.product(layer1, layer2))

    def remove_edge(self, edge: tuple):
        """
        Remove edge (u,v) from the network
        Args:
            edge: A list of how many nodes per layer
        """
        self.G.remove_edge(*edge)

    def assign_traffic_to_edges(self):
        """
        Assign traffic/weights to edges
        """
        edge_list = [edge for edge in self.G.edges]
        for edge in edge_list:
            from_edge, to_edge = edge
            self.G[from_edge][to_edge]['weight'] = self.traffic[edge]

    def plot_initial_graph(self):
        """
        Plot initial graph without weights to the edges
        """
        node_color = [self.layer_colors[data["layer"]] for v, data in self.G.nodes(data=True)]
        pos = nx.multipartite_layout(self.G, subset_key="layer")
        plt.figure(figsize=(8, 8))
        nx.draw(self.G, pos, node_color=node_color, with_labels=True, arrows=True, arrowstyle='-|>', arrowsize=20)
        plt.axis("equal")
        plt.show()

    def plot_weighted_graph(self):
        """
        Plot graph with colored edges with respect to the weights
        """
        edges, edge_color = zip(*nx.get_edge_attributes(self.G, 'weight').items())  # Get weights
        edge_cmap = LinearSegmentedColormap.from_list('gr', ["g", "r"], N=256)

        node_color = [self.layer_colors[data["layer"]] for v, data in self.G.nodes(data=True)]
        pos = nx.multipartite_layout(self.G, subset_key="layer")

        fig, ax = plt.subplots(1,figsize=(12, 8))
        nx.draw(self.G, pos, node_color=node_color,
                edge_color=edge_color, edge_cmap=edge_cmap,
                with_labels=True, arrows=True,
                arrowstyle='-|>', arrowsize=20, width=2)
        ax.axis("equal")

        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=0, vmax=np.max(self.traffic)))
        sm._A = []
        fig.colorbar(sm, label='Traffic count')
        plt.show()

    def convert_to_graph_to_dict(self):
        """
        Just convert the graph to our dictionary in dictionary format
        """
        node_list = [node for node in self.G.nodes]
        edge_list = [edge for edge in self.G.edges]

        edge_index = 0
        graph_dict = {}

        for node in node_list:
            edges_dict = {}

            while node == edge_list[edge_index][0]:
                to_edge = edge_list[edge_index][1]
                edges_dict[to_edge] = (node, node)
                if edge_index < len(edge_list) - 1:
                    edge_index += 1
                else:
                    break
            graph_dict[node] = edges_dict
        return graph_dict


def main():
    # Initialise structure of the network
    size_of_each_layer = [1, 5, 3, 1]
    traffic = np.random.randn(sum(size_of_each_layer), sum(size_of_each_layer))
    large_network = LargeNetwork(size_of_each_layer=size_of_each_layer, traffic=traffic)

    road_network = large_network.convert_to_graph_to_dict()
    pprint(road_network)

    large_network.plot_initial_graph()

    large_network.assign_traffic_to_edges()
    large_network.remove_edge((7,9))
    large_network.plot_weighted_graph()


if __name__ == '__main__':
    main()
