import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mycolorpy import colorlist as mcp


class LargeNetwork:
    def __init__(self, size_of_each_layer: list):
        self.G = None
        self.number_of_layers = len(size_of_each_layer)
        self.size_of_each_layer = size_of_each_layer
        self.layer_colors = mcp.gen_color(cmap="winter", n=self.number_of_layers)
        self.generate_multilayered_graph(*self.size_of_each_layer)

    def generate_multilayered_graph(self, *size_of_each_layer):
        extents = nx.utils.pairwise(itertools.accumulate((0,) + size_of_each_layer))
        layers = [range(start, end) for start, end in extents]
        self.G = nx.Graph()

        for (i, layer) in enumerate(layers):
            self.G.add_nodes_from(layer, layer=i)

        for layer1, layer2 in nx.utils.pairwise(layers):
            self.G.add_edges_from(itertools.product(layer1, layer2))

    def plot(self):
        color = [self.layer_colors[data["layer"]] for v, data in self.G.nodes(data=True)]
        pos = nx.multipartite_layout(self.G, subset_key="layer")
        plt.figure(figsize=(8, 8))
        nx.draw(self.G, pos, node_color=color, with_labels=False)
        plt.axis("equal")
        plt.show()


if __name__ == '__main__':
    subset_sizes = [5, 5, 4, 3, 2, 4, 4, 3]
    G = LargeNetwork(subset_sizes)
    G.plot()
