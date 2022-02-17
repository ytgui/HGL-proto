import time
import torch
import sageir
import dgl as dgl
from torch import nn
from dgl import nn as dglnn
from sageir import mp, utils
from dgl.data import rdf, reddit
from dgl.data import citation_graph as cit
from dgl.data import gnn_benchmark as bench
from common.model import GCNModel as MyGCNModel
from common.model import GATModel as MyGATModel


class DGLGCNModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.i2h = dglnn.GraphConv(
            in_features, gnn_features,
            norm='right', bias=True
        )
        self.h2o = dglnn.GraphConv(
            gnn_features, out_features,
            norm='right', bias=True
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, block, x):
        x = self.i2h(block, x)
        x = self.activation(x)
        x = self.h2o(block, x)
        return x


class DGLGATModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        self.i2h = dglnn.GATConv(
            in_features, gnn_features, n_heads
        )
        self.h2o = dglnn.GATConv(
            gnn_features, out_features, n_heads
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self, block, x):
        x = self.i2h(block, x)
        x = self.activation(x)
        x = self.h2o(block, x)
        return x


def check_dataset(dataset):
    # dataset
    divider = 10 * '-'
    print('{}[{}]{}'.format(
        divider, dataset.name, divider
    ))
    graph: dgl.DGLGraph = dataset[0]
    print('n_nodes:', graph.num_nodes())
    print('n_edges:', graph.num_edges())
    if graph.is_homogeneous:
        avg_degrees = torch.mean(
            graph.in_degrees().type(
                torch.FloatTensor
            )
        ).item()
    else:
        avg_degrees = []
        for ety in graph.canonical_etypes:
            avg_degrees.append(
                torch.mean(
                    graph.in_degrees(etype=ety).type(
                        torch.FloatTensor
                    )
                ).item()
            )
        avg_degrees = sum(avg_degrees) / len(avg_degrees)
    print('avg_degrees:', avg_degrees)


def bench_dgl_homo(dataset, model, d_hidden):
    # info
    print('[DGL] {}, {}, d_hidden={}'.format(
        type(dataset).__name__,
        model.__name__, d_hidden
    ))

    # dataset
    n_epochs = 20
    graph = dataset[0].to('cuda')
    n_nodes = graph.num_nodes()
    n_labels = dataset.num_classes
    print('n_labels:', n_labels)
    feature = graph.ndata.pop('feat')
    n_features = feature.size(-1)
    print('n_features:', n_features)

    # inputs
    feature = torch.randn(
        size=[n_nodes, n_features]
    ).to('cuda')
    model = model(
        in_features=n_features,
        gnn_features=d_hidden,
        out_features=n_labels
    ).to('cuda')

    # prewarm
    y = model(graph, feature)
    torch.sum(y).backward()

    # forward
    timing = None
    time.sleep(2.0)
    print('[FORWARD]')
    with utils.Profiler() as prof:
        for _ in range(n_epochs):
            model(graph, feature)
        timing = prof.timing() / n_epochs
    print('throughput: {:.1f}'.format(n_nodes / timing))

    # training
    timing = None
    time.sleep(2.0)
    print('[TRAINING]')
    with utils.Profiler():
        for _ in range(n_epochs):
            y = model(graph, feature)
            torch.sum(y).backward()
        timing = prof.timing() / n_epochs
    print('throughput: {:.1f}'.format(n_nodes / timing))


def bench_sageir_homo(dataset, model, d_hidden):
    # info
    print('[SAGEIR] {}, {}, d_hidden={}'.format(
        type(dataset).__name__,
        model.__name__, d_hidden
    ))

    # dataset
    n_epochs = 20
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)
    n_nodes = dglgraph.num_nodes()
    n_labels = dataset.num_classes
    print('n_labels:', n_labels)
    feature = dglgraph.ndata.pop('feat')
    n_features = feature.size(-1)
    print('n_features:', n_features)

    # inputs
    feature = torch.randn(
        size=[n_nodes, n_features]
    ).to('cuda')
    model = model(
        in_features=n_features,
        gnn_features=d_hidden,
        out_features=n_labels
    ).to('cuda')
    kwargs = dict({
        'graph': graph, 'x': feature,
        'norm': graph.right_norm()
    })

    # optimizer
    mod2ir = sageir.Module2IR()
    optimizer = sageir.Optimizer()
    dataflow = mod2ir.transform(
        model, kwargs=kwargs
    )
    dataflow = optimizer.lower(
        dataflow, kwargs=kwargs
    )
    executor = sageir.Executor()

    # prewarm
    executor.train()
    y = executor.run(
        dataflow, kwargs=kwargs
    )
    torch.sum(y).backward()

    # forward
    timing = None
    time.sleep(2.0)
    print('[FORWARD]')
    with utils.Profiler() as prof:
        for _ in range(n_epochs):
            executor.run(
                dataflow, kwargs=kwargs
            )
        timing = prof.timing() / n_epochs
    print('throughput: {:.1f}'.format(n_nodes / timing))

    # training
    timing = None
    time.sleep(2.0)
    print('[TRAINING]')
    with utils.Profiler():
        for _ in range(n_epochs):
            y = executor.run(
                dataflow, kwargs=kwargs
            )
            torch.sum(y).backward()
        timing = prof.timing() / n_epochs
    print('throughput: {:.1f}'.format(n_nodes / timing))


def main():
    for dataset in [
        cit.CoraGraphDataset,
        bench.CoraFullDataset,
        cit.PubmedGraphDataset,
        reddit.RedditDataset,
        #
        rdf.AIFBDataset, rdf.MUTAGDataset,
        rdf.BGSDataset, rdf.AMDataset
    ]:
        dataset = dataset(
            verbose=False
        )
        check_dataset(dataset)
        for dgl_model, sageir_model in [
            (DGLGCNModel, MyGCNModel),
            (DGLGATModel, MyGATModel)
        ]:
            for d_hidden in [16, 32, 64]:
                bench_dgl_homo(
                    dataset=dataset,
                    model=dgl_model, d_hidden=d_hidden,
                )
                time.sleep(2.0)
                #
                bench_sageir_homo(
                    dataset=dataset,
                    model=sageir_model, d_hidden=d_hidden,
                )
                time.sleep(2.0)


if __name__ == "__main__":
    """
    + end-to-end speedup
    + memory allocation, reclamation
    + kernel counting and portions
    """
    main()
