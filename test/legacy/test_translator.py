import torch
import sageir
import dgl as dgl
from torch import nn
from sageir import mp
from dgl.data import CoraGraphDataset


"""
class GATLayer(nn.Module):
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
        h = self.linear_v(x)
        q = self.linear_q(h)
        k = self.linear_k(h)
        self.graph.src_node['u'] = x
        self.graph.src_node['el'] = q
        self.graph.dst_node['er'] = k

        # gat attention
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


class GATModel(nn.Module):
    def __init__(self, graph, in_features, hidden_features, out_features):
        super().__init__()
        #
        self.layer = nn.Sequential(
            GATLayer(graph, in_features, hidden_features),
            nn.ELU(),
            nn.Dropout(),
            GATLayer(graph, hidden_features, out_features)
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
"""


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #
        self.linear_q = nn.Linear(out_features, 1)
        self.linear_k = nn.Linear(out_features, 1)
        self.linear_v = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph: mp.Graph, x: torch.Tensor):
        # different to bert
        h = self.linear_v(x)
        q = self.linear_q(h)
        k = self.linear_k(h)
        graph.src_node['u'] = h
        graph.src_node['el'] = q
        graph.dst_node['er'] = k

        # gat attention
        graph.message_func(mp.F.u_add_v('el', 'er', 'e'))
        graph.edge['coeff'] = self.leaky_relu(graph.edge['e'])
        """
        graph.message_func(mp.F.edge_softmax('coeff', 'attn'))
        graph.message_func(mp.F.u_mul_e('u', 'attn', 'm'))
        graph.reduce_func(mp.F.aggregate_sum('m', 'v'))
        return graph.dst_node['v']
        """
        return graph.edge['coeff']


def test():
    dataset = CoraGraphDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)

    #
    label = dglgraph.ndata.pop(
        'label'
    ).type(torch.IntTensor)
    feature = dglgraph.ndata.pop(
        'feat'
    ).type(torch.FloatTensor)
    n_nodes = dglgraph.num_nodes()
    n_labels = dataset.num_classes
    n_features = feature.size(1)

    #
    d_hidden = 16
    layer = GATLayer(
        in_features=n_features,
        out_features=d_hidden
    ).to('cuda')
    feature = feature.to('cuda')

    #
    mod2ir = sageir.Module2IR()
    dataflow = mod2ir.transform(
        layer, kwargs={
            'graph': graph,
            'x': feature
        }
    )
    sageir.Printer().dump(dataflow)
    a = 0


if __name__ == "__main__":
    test()
