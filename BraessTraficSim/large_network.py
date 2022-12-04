import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mycolorpy import colorlist as mcp
from pprint import pprint
from matplotlib.colors import LinearSegmentedColormap


class LargeNetwork:
    def __init__(self, size_of_each_layer: list):
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

        self.traffic_in_edges = None
        self.traffic_parameters = {}  # dict {edges: (a,b)}
        self.traffic_parameters_original = None
        self.generate_traffic_parameters()

    def generate_traffic_parameters(self):
        """
        Generate traffic parameters a,b to each edge
        """
        edge_list = [edge for edge in self.G.edges]
        for edge in edge_list:
            self.traffic_parameters[edge] = (100 * np.random.rand(), 100 * np.random.rand())
        self.traffic_parameters_original = self.traffic_parameters.copy()

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

    def add_edge(self, edge: tuple):
        """
        Add edge (u,v) to the network
        If the edge already exist do nothing
        Args:
            edge: A list of how many nodes per layer
        """
        edge_list = [edge for edge in self.G.edges]
        if edge not in edge_list:
            self.G.add_edge(*edge)
            # Assign from random parameters or (0,0)
            # self.traffic_parameters[edge] = self.traffic_parameters_original[edge]
            self.traffic_parameters[edge] = (0, 0)
        else:
            print('The edge you want to add already exists')

    def remove_edge(self, edge: tuple):
        """
        Remove edge (u,v) from the network
        If the edge does not exist don't do anything
        Args:
            edge: A list of how many nodes per layer
        """
        edge_list = [edge for edge in self.G.edges]
        if edge in edge_list:
            self.G.remove_edge(*edge)
            del self.traffic_parameters[edge]
        else:
            print('The edge that you ant to remove does not exist')

    def assign_traffic_to_edges(self, traffic_in_edges):
        """
        Assign traffic/weights to edges
            :param traffic_in_edges: Traffic in each edge
            :type traffic_in_edges: np.ndarray
        """
        self.traffic_in_edges = traffic_in_edges
        edge_list = [edge for edge in self.G.edges]
        for edge in edge_list:
            from_edge, to_edge = edge
            self.G[from_edge][to_edge]['weight'] = traffic_in_edges[edge]

        # This is just for the debug, printing the weights for eah edge
        # for item in nx.get_edge_attributes(self.G, 'weight').items(): # Get weights
        #   print(item)

    def plot_initial_graph(self):
        """
        Plot initial graph without weights to the edges
        """
        node_color = [self.layer_colors[data["layer"]] for v, data in self.G.nodes(data=True)]
        pos = nx.multipartite_layout(self.G, subset_key="layer")
        plt.figure(figsize=(8, 8))
        nx.draw(self.G, pos, node_color=node_color, with_labels=True, arrows=True, arrowstyle='-|>', arrowsize=20)
        plt.axis("equal")
        plt.show(block=False)

    def plot_weighted_graph(self, driver_probability):
        """
        Plot graph with colored edges with respect to the weights
        """
        edges, edge_color = zip(*nx.get_edge_attributes(self.G, 'weight').items())  # Get weights
        edge_cmap = LinearSegmentedColormap.from_list('gr', ["g", "r"], N=256)

        node_color = [self.layer_colors[data["layer"]] for v, data in self.G.nodes(data=True)]
        pos = nx.multipartite_layout(self.G, subset_key="layer")

        fig, ax = plt.subplots(1, figsize=(10, 6))
        # Draw network with node labels and weighted arrows
        nx.draw(self.G, pos, node_color=node_color,
                edge_color=edge_color, edge_cmap=edge_cmap,
                with_labels=True, arrows=True,
                arrowstyle='-|>', arrowsize=20, width=2)

        # Draw weights between edges
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        ax.axis("equal")
        ax.set_title(f'p = {driver_probability:.2f}')

        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=0, vmax=np.max(self.traffic_in_edges)))
        sm._A = []
        fig.colorbar(sm, label='Traffic count')
        plt.show(block=False)

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
                edge_key = edge_list[edge_index]

                # Add traffic parameters to the edge
                a, b = self.traffic_parameters[edge_key]
                edges_dict[to_edge] = (a, b)
                if edge_index < len(edge_list) - 1:
                    edge_index += 1
                else:
                    break
            graph_dict[node] = edges_dict
        return graph_dict


def main():
    # Initialise structure of the network
    size_of_each_layer = [1, 3, 4, 3, 1]
    large_network = LargeNetwork(size_of_each_layer=size_of_each_layer)

    road_network = large_network.convert_to_graph_to_dict()
    pprint(road_network)
    large_network.plot_initial_graph()

    traffic_in_edges = np.random.rand(sum(size_of_each_layer), sum(size_of_each_layer))
    large_network.assign_traffic_to_edges(traffic_in_edges)


if __name__ == '__main__':
    main()
