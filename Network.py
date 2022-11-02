import random

import networkx as nx


class Network:
    def __init__(self, N):
        self.G = nx.Graph()
        self.N = N
        self.H = nx.path_graph(N)
        self.L = {}
        for i in range(N):
            for j in range(i + 1, N):
                self.rand_add_edge(i, j)

        for edge in self.G.edges():
            a, b = edge
            self.G[a][b]['R'] = 10

        self.set_all_l()

    def rand_add_edge(self, vi, vj, p=0.2):
        rd = random.random()
        if rd < p:
            self.G.add_edge(vi, vj)

    def get_net(self):
        return self.G

    def get_l(self, a, b):
        sum_L = 0
        try:
            path = nx.dijkstra_path(self.G, a, b)
            for a, b in zip(path[:-1], path[1:]):
                sum_L += self.G[a][b]['R']
            return sum_L
        except:
            return -1

    def set_all_l(self):

        for i in range(self.N):
            for j in range(self.N):
                self.L[(i, j)] = 1e5

        for a, b in self.G.edges:
            self.L[(a, b)] = self.G[a][b]['R']
            self.L[(b, a)] = self.G[b][a]['R']

        for k in range(self.N):
            for i in range(self.N):
                for j in range(self.N):
                    self.L[(i, j)] = max(self.L[(i, j)], self.L[(i, k)] + self.L[(k, j)])
