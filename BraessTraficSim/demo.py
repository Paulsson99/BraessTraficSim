import numpy as np

class TraficAnt:
    def __init__(self, graph, N):
        self.graph = graph
        self.N = N
        self.tau = np.ones((len(graph), len(graph)))
        self.alpha = 2
        self.trafic_count = dict()

    def evaluation_edge(self, a, b, c, u):
        return a * u * u + b * u + c

    def evaluate_route(self, route, trafic):
        sum = 0

        for i, j in zip(route[:-1], route[1:]):
            edge = self.graph[i][j]
            edge_trafic = trafic[i][j]
            sum += self.evaluation_edge(*edge, edge_trafic)

        return sum

    def calc_route(self):
        route = [1]
        last = route[-1]

        while(last != 4):
            next = self.next_node(last)
            try:
                self.trafic_count[last][next] += 1
            except KeyError:
                self.trafic_count[last] = dict()
                self.trafic_count[last][next] = 1

            route.append(next)
            last = route[-1]

        return route

    def next_node(self, nodeI, route):
        pos_next = self.graph[nodeI].keys()
        unseen_nodes = pos_next - route # is this possible?
        prob = self.tau[nodeI, unseen_nodes] ** self.alpha
        prob /= sum(prob)
        return np.random.choice(unseen_nodes, p = prob)

    def run(self):
        routes = [self.calc_route() for i in range(self.N)]
        


