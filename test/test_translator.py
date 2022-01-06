import torch
import dgl as dgl
import networkx as nx
from torch import nn


class F:
    @classmethod
    def u_add_v(cls, u, v, e):
        pass

    @classmethod
    def u_mul_e(cls, u, e, v):
        pass

    @classmethod
    def edge_softmax(cls, e1, e2):
        pass

    @classmethod
    def aggregate_sum(cls, e, v):
        pass


class MPGraph:
    def __init__(self, adj: torch.Tensor):
        self.adj = adj
        self.edge = dict()
        self.src_node = dict()
        self.dst_node = dict()

    def num_nodes(self) -> int:
        return self.adj.size(0)

    def out_degrees(self) -> torch.Tensor:
        return torch.sum(self.adj, dim=1) 

    def message_func(self, func):
        # message function defined on each edge to generate a message
        # by combining the edge feature with the features of its
        # incident nodes
        a = 0

    def reduce_func(self, func):
        # update function defined on each node to update the node
        # feature by aggregating its incoming messages using the
        # reduce function
        a = 0


class MPGATLayer(nn.Module):
    def __init__(self, graph: MPGraph, in_features, out_features):
        super().__init__()
        #
        self.graph = graph
        self.linear_q = nn.Linear(out_features, 1)
        self.linear_k = nn.Linear(out_features, 1)
        self.linear_v = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        # different to bert
        x = self.linear_v(x)
        q = self.linear_q(x)
        k = self.linear_k(x)
        self.graph.src_node['u'] = x
        self.graph.src_node['el'] = q
        self.graph.dst_node['er'] = k

        # addition attention
        self.graph.message_func(
            F.u_add_v('el', 'er', 'e')
        )
        coeff = self.leaky_relu(
            self.graph.edge['e']
        )
        self.graph.edge['coeff'] = coeff
        self.graph.message_func(
            F.edge_softmax('coeff', 'attn')
        )
        self.graph.message_func(
            F.u_mul_e('u', 'attn', 'm')
        )
        self.graph.reduce_func(
            F.aggregate_sum('m', 'v')
        )
        return self.graph.dst_node['v']


class GATConv(nn.Module):
    def __init__(self, graph, in_features, hidden_features, out_features):
        super().__init__()
        #
        self.layer = nn.Sequential(
            MPGATLayer(graph, in_features, hidden_features),
            nn.ELU(),
            nn.Dropout(),
            MPGATLayer(graph, hidden_features, out_features)
        )

    def forwrad(self, x):
        return self.layer(x)


class Translator:
    def __init__(self) -> None:
        pass


def test():
    n_nodes = 256
    density = 0.05
    nx_graph = nx.erdos_renyi_graph(
        n=n_nodes, p=density
    )
    dgl_graph = dgl.DGLGraph(nx_graph)
    adjacency = torch.FloatTensor(
        nx.adjacency_matrix(
            nx_graph
        ).todense()
    )
    mp_graph = MPGraph(adjacency)

    #
    a = 0


if __name__ == "__main__":
    # gat: message passing impl
    # r-gat: migrate to heterograph
    # translator: ir implementation detail
    # draft refine: bottom up system, limited optimization, heterograph first
    test()
