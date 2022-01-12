import torch
import sageir
from torch import nn
from sageir import ir
from dgl import nn as dglnn
from dgl.data import CoraGraphDataset
from dgl.data.rdf import AIFBDataset, MUTAGDataset, AMDataset
from tqdm import tqdm


class GCNLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.fc = nn.Linear(
            in_features, out_features
        )

    def forward(self,
                block: sageir.Block,
                x: torch.Tensor):
        x = self.fc(x)
        x = sageir.gspmm(
            block.adj_sparse,
            block.rev_sparse,
            x
        )
        return x


class GCNModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.i2h = GCNLayer(in_features, gnn_features)
        self.h2o = GCNLayer(gnn_features, out_features)

    def forward(self,
                block: sageir.Block,
                x: torch.Tensor):
        x = self.i2h(block, x)
        x = self.h2o(block, x)
        return x


def check_homo():
    dataset = CoraGraphDataset(
        verbose=False
    )
    g = dataset[0].to('cuda')
    block = sageir.from_dglgraph(g)

    #
    label = g.ndata.pop(
        'label'
    ).type(torch.IntTensor)
    feature = g.ndata.pop(
        'feat'
    ).type(torch.FloatTensor)
    n_nodes = g.num_nodes()
    n_labels = dataset.num_classes
    n_features = feature.size(1)

    #
    d_hidden = 16
    model = GCNModel(
        in_features=n_features,
        gnn_features=d_hidden,
        out_features=n_labels
    ).to('cuda')
    label = label.to('cuda')
    feature = feature.to('cuda')

    #
    mod2ir = sageir.Module2IR()
    dag = mod2ir.transform(
        model, args=(block, feature)
    )
    a = 0


def check_hetero():
    dataset = MUTAGDataset(
        verbose=True
    )
    graph = dataset[0].to('cuda')
    pred_cat = dataset.predict_category
    print('predict_category:', pred_cat)

    #
    features = {}
    for nty in graph.ntypes:
        a = 0
    x = ir.OpTensor(size=[])


def test():
    for _ in tqdm(range(256)):
        check_homo()
        # check_hetero()


if __name__ == "__main__":
    test()
