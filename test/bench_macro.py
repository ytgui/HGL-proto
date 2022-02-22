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
from common.model import GCNModel, GATModel, RGATModel


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
            in_features, gnn_features, n_heads,
            activation=None, bias=True
        )
        self.h2o = dglnn.GATConv(
            gnn_features, out_features, n_heads,
            activation=None, bias=True
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self, block, x):
        x = torch.mean(
            self.i2h(block, x),
            dim=1
        )
        x = self.activation(x)
        x = torch.mean(
            self.h2o(block, x),
            dim=1
        )
        return x


class DGLREmbedding(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 embedding_dim: int):
        nn.Module.__init__(self)
        self.embeds = nn.ModuleDict({
            nty: nn.Embedding(
                g.num_nodes(nty),
                embedding_dim
            )
            for nty in g.ntypes
        })

    def forward(self, block):
        return {
            nty: self.embeds[
                nty
            ].weight
            for nty in sorted(list(block.ntypes))
        }


class DGLRGATModel(nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        #
        self.embed = DGLREmbedding(
            g, embedding_dim=in_features
        )
        self.i2h = dglnn.HeteroGraphConv(
            {
                ety: dglnn.GATConv(
                    in_features, gnn_features, n_heads,
                    activation=None, bias=True
                )
                for ety in sorted(list(g.etypes))
            }, aggregate='sum'
        )
        self.h2o = dglnn.HeteroGraphConv(
            {
                ety: dglnn.GATConv(
                    gnn_features, out_features, n_heads,
                    activation=None, bias=True
                )
                for ety in sorted(list(g.etypes))
            }, aggregate='sum'
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, block):
        xs = self.embed(block)
        hs = self.i2h(block, xs)
        hs = {
            k: self.activation(
                torch.mean(h, dim=1)
            )
            for k, h in hs.items()
        }
        hs = self.h2o(block, hs)
        hs = {
            k: torch.mean(h, dim=1)
            for k, h in hs.items()
        }
        return hs


class BenchMethods:
    @staticmethod
    def _check_dataset(dataset):
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
            print(
                'meta-paths:',
                len(graph.canonical_etypes)
            )
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

    @staticmethod
    def _bench_dgl_homo(dataset, model, d_hidden):
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
        feature = graph.ndata.pop('feats')
        n_features = feature.size(-1)
        print('n_features:', n_features)

        # inputs
        feature = torch.randn(
            size=[n_nodes, n_features]
        ).to('cuda')
        gradient = torch.ones(
            [n_nodes, n_labels]
        ).to('cuda')
        model = model(
            in_features=n_features,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')

        # prewarm
        y = model(graph, feature)
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler() as prof:
            for _ in range(n_epochs):
                y = model(graph, feature)
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_nodes / timing))

    @staticmethod
    def _bench_dgl_hetero(dataset, model, d_hidden):
        # info
        print('[DGL] {}, {}, d_hidden={}'.format(
            type(dataset).__name__,
            model.__name__, d_hidden
        ))

        # dataset
        n_epochs = 20
        graph = dataset[0].to('cuda')
        n_nodes = graph.num_nodes()
        print('n_nodes:', n_nodes)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        category = dataset.predict_category
        print('predict_category:', category)

        # inputs
        gradient = torch.ones([
            graph.num_nodes(category),
            n_labels
        ]).to('cuda')
        model = model(
            g=graph,
            in_features=d_hidden,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')

        # prewarm
        y = model(graph)[category]
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler() as prof:
            for _ in range(n_epochs):
                y = model(graph)[category]
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_nodes / timing))

    @staticmethod
    def _bench_sageir_homo(dataset, model, d_hidden):
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
        gradient = torch.ones(
            [n_nodes, n_labels]
        ).to('cuda')
        model = model(
            in_features=n_features,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')
        kwargs = dict({
            'graph': graph, 'x': feature
        })
        if isinstance(model, GCNModel):
            kwargs['norm'] = graph.right_norm()

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
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler() as prof:
            for _ in range(n_epochs):
                y = executor.run(
                    dataflow, kwargs=kwargs
                )
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_nodes / timing))

    @staticmethod
    def _bench_sageir_hetero(dataset, model, d_hidden):
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
        print('n_nodes:', n_nodes)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        category = dataset.predict_category
        print('predict_category:', category)

        # inputs
        n_heads = 2
        gradient = torch.ones([
            dglgraph.num_nodes(category),
            n_labels
        ]).to('cuda')
        model = model(
            hgraph=graph,
            in_features=d_hidden,
            gnn_features=d_hidden,
            out_features=n_labels,
            n_heads=n_heads
        ).to('cuda')
        node_indices = {
            nty: torch.linspace(
                0, num - 1, num,
                dtype=torch.int64
            ).to('cuda')
            for nty, num in graph.nty2num.items()
        }
        kwargs = dict({
            'hgraph': graph,
            'xs': node_indices
        })

        # optimizer
        mod2ir = sageir.Module2IR()
        optimizer = sageir.Optimizer()
        dataflow = mod2ir.transform(
            model, kwargs=kwargs
        )[category]
        dataflow = optimizer.lower(
            dataflow, kwargs=kwargs
        )
        executor = sageir.Executor()

        # prewarm
        executor.train()
        y = executor.run(
            dataflow, kwargs=kwargs
        )
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler() as prof:
            for _ in range(n_epochs):
                y = executor.run(
                    dataflow, kwargs=kwargs
                )
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_nodes / timing))


class Benchmark(BenchMethods):
    HOM_DATASETS = [
        cit.CoraGraphDataset,
        bench.CoraFullDataset,
        # cit.PubmedGraphDataset,
        # reddit.RedditDataset
    ]
    HET_DATASETS = [
        rdf.AIFBDataset, rdf.MUTAGDataset,
        # rdf.BGSDataset, rdf.AMDataset
    ]
    HOM_CMP_MODELS = [
        (DGLGATModel, GATModel),
        (DGLGCNModel, GCNModel)
    ]
    HET_CMP_MODELS = [
        (DGLRGATModel, RGATModel)
    ]

    def dataset_info(self):
        for dataset in self.HOM_DATASETS:
            dataset = dataset(
                verbose=False
            )
            self._check_dataset(dataset)
        for dataset in self.HET_DATASETS:
            dataset = dataset(
                verbose=False
            )
            self._check_dataset(dataset)

    def bench_homogenous(self):
        for dataset in self.HOM_DATASETS:
            dataset = dataset(
                verbose=False
            )
            for dgl_model, sageir_model in \
                    self.HOM_CMP_MODELS:
                for d_hidden in [16]:
                    self._bench_dgl_homo(
                        dataset=dataset,
                        model=dgl_model,
                        d_hidden=d_hidden,
                    )
                    time.sleep(2.0)
                    self._bench_sageir_homo(
                        dataset=dataset,
                        model=sageir_model,
                        d_hidden=d_hidden,
                    )
                    time.sleep(2.0)
                    return

    def bench_heterogenous(self):
        for dataset in self.HET_DATASETS:
            dataset = dataset(
                verbose=False
            )
            for dgl_model, sageir_model in \
                    self.HET_CMP_MODELS:
                for d_hidden in [16]:
                    self._bench_dgl_hetero(
                        dataset=dataset,
                        model=dgl_model,
                        d_hidden=d_hidden,
                    )
                    time.sleep(2.0)
                    self._bench_sageir_hetero(
                        dataset=dataset,
                        model=sageir_model,
                        d_hidden=d_hidden,
                    )
                    time.sleep(2.0)


def main():
    bench = Benchmark()
    # bench.dataset_info()
    bench.bench_heterogenous()


if __name__ == "__main__":
    """
    + end-to-end speedup
    + memory allocation, reclamation
    + kernel counting and portions
    """
    main()
