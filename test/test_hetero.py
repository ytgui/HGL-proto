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
                feature: torch.Tensor):
        h = self.fc(feature)
        h = sageir.gspmm(block, h)
        return h


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
                feature: torch.Tensor):
        x = self.i2h(
            block, feature
        )
        x = self.h2o(block, x)
        return x


class HeteroGraphConv(nn.Module):
    def __init__(self, convs: dict):
        nn.Module.__init__(self)
        self._convs = nn.ModuleDict(convs)

    def forward(self, hblock, features):
        counts = {}
        outputs = {}
        for (sty, ety, dty), block \
                in hblock:
            res = self._convs[ety](
                block, features[sty]
            )
            if dty not in outputs:
                counts[dty] = 0
                outputs[dty] = torch.zeros(
                    size=res.size(),
                    device=res.device
                )
            counts[dty] += 1
            outputs[dty] = outputs[dty] + res
        for dty, res in outputs.items():
            outputs[dty] = res / counts[dty]
        return outputs


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
    dataflow = mod2ir.transform(
        model, kwargs={
            'block': block,
            'feature': feature
        }
    )
    sageir.Printer().dump(dataflow)

    #
    executor = sageir.Executor()
    # y = executor.run(dataflow)
    a = 0


def check_hetero():
    dataset = MUTAGDataset(
        verbose=True
    )
    g = dataset[0].to('cuda')
    hblock = sageir.from_dglgraph(g)
    n_labels = dataset.num_classes
    pred_cat = dataset.predict_category
    print('predict_category:', pred_cat)

    #
    d_hidden = 16
    model = HeteroGraphConv({
        ety: GCNLayer(
            in_features=d_hidden,
            out_features=d_hidden
        )
        for ety in hblock.etypes
    }).to('cuda')
    label = g.ndata.pop(
        'labels'
    )[pred_cat].type(
        torch.IntTensor
    ).to('cuda')
    features = {
        nty: torch.randn(size=[
            g.number_of_nodes(nty),
            d_hidden
        ]).to('cuda')
        for nty in g.ntypes
    }

    #
    mod2ir = sageir.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs={
            'hblock': hblock,
            'features': features
        }
    )
    sageir.Printer().dump(dataflow)

    #
    a = 0


def test():
    for _ in tqdm(range(256)):
        # check_homo()
        check_hetero()


if __name__ == "__main__":
    test()
