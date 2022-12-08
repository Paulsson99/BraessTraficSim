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
            :param size_of_each_layer: A list of how many nodes per layer
        """

        self.G = None
        self.number_of_layers = len(size_of_each_layer)
        self.size_of_each_layer = size_of_each_layer
        self.layer_colors = mcp.gen_color(cmap="cool", n=self.number_of_layers)
        self.generate_multilayered_graph(*self.size_of_each_layer)

        self.traffic_in_edges = None
        self.traffic_parameters = {}  # dict {edges: (delta, epsilon, c)}

    def generate_multilayered_graph(self, *size_of_each_layer: list):
        """
        Generate a multilayered graph
            :param size_of_each_layer: A list of how many nodes per layer
        """
        extents = nx.utils.pairwise(itertools.accumulate((0,) + size_of_each_layer))
        layers = [range(start, end) for start, end in extents]
        self.G = nx.DiGraph()  # Graph vs DiGraph: Graph (u,v)==(v,u), DiGraph (u,v) != (v,u)

        for (i, layer) in enumerate(layers):
            self.G.add_nodes_from(layer, layer=i)

        for layer1, layer2 in nx.utils.pairwise(layers):
            self.G.add_edges_from(itertools.product(layer1, layer2))

    def add_edge(self, edge: tuple, davidson_parameters:tuple):
        """
        Add edge (u,v) to the network and davidson parameter to this edge
        If the edge already exist do nothing
            :param edge: A tuple of how many nodes per layer
            :param davidson_parameters: (t0, epsilon, c)
        """
        if not self.G.has_edge(*edge):
            self.G.add_edge(*edge)
            self.traffic_parameters[edge] = davidson_parameters
        else:
            print(f'\nThe edge {edge} that you want to add already exists')

    def remove_edge(self, edge: tuple):
        """
        Remove edge (u,v) from the network
        If the edge does not exist don't do anything
        :param
            edge: A tuple of how many nodes per layer
        """
        if self.G.has_edge(*edge):
            self.G.remove_edge(*edge)
            del self.traffic_parameters[edge]
        else:
            print(f'\nThe edge {edge} that you ant to remove does not exist')

    def assign_traffic_parameters(self, min_max_road_parameters: dict):
        """
        Assign traffic parameters (t0, epsilon, c) to each edge
            :param min_max_road_parameters:
        """
        edge_list = [edge for edge in self.G.edges]
        for edge in edge_list:
            min_parameter_list = [min_max_road_parameters['min_t0'], min_max_road_parameters['min_eps'], min_max_road_parameters['min_c']]
            max_parameter_list = [min_max_road_parameters['max_t0'], min_max_road_parameters['max_eps'], min_max_road_parameters['max_c']]
            self.traffic_parameters[edge] = tuple(np.random.uniform(min_parameter_list, max_parameter_list))

    def assign_traffic_to_edges(self, traffic_in_edges: np.ndarray):
        """
        Assign traffic/weights to edges
            :param traffic_in_edges: Traffic in each edge
        """
        self.traffic_in_edges = traffic_in_edges
        edge_list = [edge for edge in self.G.edges]
        for edge in edge_list:
            from_edge, to_edge = edge
            self.G[from_edge][to_edge]['weight'] = traffic_in_edges[edge]

    def plot_initial_graph(self):
        """
        Plot initial graph without weights to the edges
        """

        # Extract edges that are bidirectional
        curved_edges = [edge for edge in self.G.edges() if reversed(edge) in self.G.edges()]
        straight_edges = list(set(self.G.edges()) - set(curved_edges))

        node_color = [self.layer_colors[data["layer"]] for v, data in self.G.nodes(data=True)]
        pos = nx.multipartite_layout(self.G, subset_key="layer")

        fig, ax = plt.subplots(1)

        # Draw node labels
        nx.draw_networkx_labels(self.G, pos, ax=ax,
                                font_size=12)
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, ax=ax,
                               node_color=node_color, node_size=400)
        # Draw one directed edges
        nx.draw_networkx_edges(self.G, pos,
                               edgelist=straight_edges, edge_color='black',
                               arrows=True, arrowstyle='-|>', arrowsize=16,
                               width=2, connectionstyle='arc3, rad = 0.0')
        # If there are som bidirectional edges, we draw them as curved
        if curved_edges:
            nx.draw_networkx_edges(self.G, pos,
                                   edgelist=curved_edges, edge_color='black',
                                   arrows=True, arrowstyle='-|>', arrowsize=20,
                                   width=2, connectionstyle='arc3, rad = 0.2')
            # Draw weights as labels for all edges
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, label_pos=0.4)
        ax.axis('off')
        plt.show(block=False)

    def plot_weighted_graph(self, driver_probability):
        """
        Plot graph with colored edges with respect to the weights
        """
        # Extract edges that are bidirectional
        curved_edges = [edge for edge in self.G.edges() if reversed(edge) in self.G.edges()]
        straight_edges = list(set(self.G.edges()) - set(curved_edges))

        node_color = [self.layer_colors[data["layer"]] for v, data in self.G.nodes(data=True)]
        edge_cmap = LinearSegmentedColormap.from_list(name='grey-black', colors=["grey", "black"], N=256)

        edges, edge_weights = zip(*nx.get_edge_attributes(self.G, 'weight').items())  # Get weights
        pos = nx.multipartite_layout(self.G, subset_key="layer")

        # Dictionary with edges as keys and weights as values
        edge_weight_dict = dict(zip(edges, edge_weights))
        straight_edges_weights = [edge_weight_dict[edge] for edge in straight_edges]

        fig, ax = plt.subplots(1)

        # Draw node labels
        nx.draw_networkx_labels(self.G, pos, ax=ax,
                                font_size=12)
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, ax=ax,
                               node_color=node_color, node_size=400)
        # Draw one directed edges
        nx.draw_networkx_edges(self.G, pos,
                               edgelist=straight_edges, edge_color=straight_edges_weights, edge_cmap=edge_cmap,
                               arrows=True, arrowstyle='-|>', arrowsize=16,
                               width=3 * np.asarray(straight_edges_weights) / np.max(edge_weights),
                               connectionstyle='arc3, rad = 0.0')

        # Draw weights labels for straight edges
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=dict(zip(straight_edges, straight_edges_weights)),
                                     label_pos=0.4)

        # If there are som bidirectional edges, we draw them as curved
        if curved_edges:
            curved_edges_weights = [edge_weight_dict[edge] for edge in curved_edges]
            nx.draw_networkx_edges(self.G, pos,
                                   edgelist=curved_edges, edge_color=curved_edges_weights, edge_cmap=edge_cmap,
                                   arrows=True, arrowstyle='-|>', arrowsize=15,
                                   width=3 * np.asarray(curved_edges_weights) / np.max(edge_weights),
                                   connectionstyle='arc3, rad = 0.2')
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=dict(zip(curved_edges, curved_edges_weights)),
                                         label_pos=0.25)

        ax.axis("off")
        ax.set_title(f'p = {driver_probability:.2f}')

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
            #  Looping through the tuples with the first part equals to node
            while node == edge_list[edge_index][0]:
                to_edge = edge_list[edge_index][1]
                edge_key = edge_list[edge_index]

                # Add traffic parameters to the edge
                edges_dict[to_edge] = self.traffic_parameters[edge_key]
                if edge_index < len(edge_list) - 1:
                    edge_index += 1
                else:
                    break
            graph_dict[node] = edges_dict
        return graph_dict


def main():
    # Initialise structure of the network
    size_of_each_layer = [1, 2, 1]
    large_network = LargeNetwork(size_of_each_layer=size_of_each_layer)
    # large_network.assign_traffic_to_edges(np.array([[0, 't=T/100', 't=45', 0],
    #                                                 [10, 11, 12, 't=45'],
    #                                                 [20, 21, 22, 't=T/100'],
    #                                                 [30, 31, 32, 33]]))

    #  pprint(large_network.convert_to_graph_to_dict())
    #  large_network.plot_initial_graph()
    large_network.add_edge((1, 2))
    large_network.add_edge((2, 1))
    # pprint(large_network.convert_to_graph_to_dict())
    # large_network.plot_initial_graph()
    #
    traffic_in_edges = np.round(np.random.rand(sum(size_of_each_layer), sum(size_of_each_layer)), 3)
    large_network.assign_traffic_to_edges(traffic_in_edges)
    large_network.plot_weighted_graph(driver_probability=0.1)
    plt.show()


if __name__ == '__main__':
    main()
