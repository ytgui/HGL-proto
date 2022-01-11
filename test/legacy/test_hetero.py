import os
import torch
import dgl as dgl
from torch import nn
from dgl import nn as dglnn
from ogb import nodeproppred


class REmbedding(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 embed_dim: int,
                 excludes: list):
        nn.Module.__init__(self)
        #
        self.g = g
        self.embeds = nn.ModuleDict()
        for nty in g.ntypes:
            if nty in excludes:
                continue
            n = g.number_of_nodes(nty)
            assert nty not in self.embeds
            self.embeds[nty] = nn.Embedding(n, embed_dim)

    def forward(self, x):
        features = {
            nty: self.embeds[nty]
            for nty in self.embeds
        }
        features['paper'] = x
        return features


class RGCNLayer(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 in_features: int,
                 out_features: int,
                 activation=None,
                 dropout=None):
        nn.Module.__init__(self)
        #
        self.g = g
        self.convs = dglnn.HeteroGraphConv(
            {
                ety: dglnn.GraphConv(
                    in_features,
                    out_features
                ) for ety in sorted(list(g.etypes))
            },
            aggregate='sum'
        )
        self.activation = activation
        self.dropout = dropout

    def forward(self, features):
        hs = self.convs(self.g, features)
        return hs


class RGCNModel(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.embed = REmbedding(
            g,
            embed_dim=in_features,
            excludes=['paper']
        )
        self.i2h = RGCNLayer(
            g,
            in_features,
            gnn_features,
            activation=nn.ReLU(),
            dropout=nn.Dropout(0.1)
        )
        self.h2o = RGCNLayer(
            g,
            gnn_features,
            out_features,
            activation=None,
            dropout=None
        )

    def forward(self, xs):
        xs = self.embed(xs)
        a = 0


def test():
    home = os.getenv('HOME')
    dataset = nodeproppred.DglNodePropPredDataset(
        root=home + '/ogb_dataset', name='ogbn-mag'
    )
    graph = dataset.graph[0]
    split_idx = dataset.get_idx_split()
    labels = dataset.labels['paper'].flatten()
    features = graph.ndata.pop('feat')['paper']

    #
    n_hidden = 64
    n_labels = dataset.num_classes
    n_features = features.size(-1)
    print('n_labels:', n_labels)
    print('n_features:', n_features)
    for nty in graph.ntypes:
        print('num_nodes:', nty, graph.number_of_nodes(nty))
    for ety in graph.etypes:
        print('num_edges:', ety, graph.number_of_edges(ety))
    print('features:', features.size())

    #
    model = RGCNModel(
        graph,
        in_features=n_features,
        gnn_features=n_hidden,
        out_features=n_labels
    )
    out = model(features)
    a = 0


if __name__ == "__main__":
    test()
