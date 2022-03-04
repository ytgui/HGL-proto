import time
import torch
import sageir
import dgl as dgl
from sageir import mp, utils
from dgl.data import rdf, reddit
from dgl.data import citation_graph as cit
from dgl.data import gnn_benchmark as bench
from torch_geometric import datasets as pygds
from common.model import GCNModel, GATModel, RGATModel
from common.dglmodel import DGLGCNModel, DGLGATModel, DGLRGATModel
from common.pygmodel import PyGGCNModel


class BenchMethods:
    @staticmethod
    def _check_dataset(dataset):
        # dataset
        divider = 10 * '-'
        print('{}[{}]{}'.format(
            divider, dataset.name, divider
        ))
        graph: dgl.DGLGraph = dataset[0]
        if graph.is_homogeneous:
            avg_degrees = torch.mean(
                graph.in_degrees().type(
                    torch.FloatTensor
                )
            ).item()
            print('n_nodes:', graph.num_nodes())
            print('n_edges:', graph.num_edges())
            print('avg_degrees:', avg_degrees)
        else:
            n_nodes = 0
            n_edges = 0
            avg_degrees = []
            print(
                'meta-paths:',
                len(graph.canonical_etypes)
            )
            for sty, ety, dty in graph.canonical_etypes:
                avg_degrees.append(
                    torch.mean(
                        graph.in_degrees(
                            etype=(sty, ety, dty)
                        ).type(
                            torch.FloatTensor
                        )
                    ).item()
                )
                n_nodes += graph.num_dst_nodes(dty)
                n_edges += graph.num_edges((sty, ety, dty))
            print('n_nodes:', n_nodes)
            print('n_edges:', n_edges)
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
        print('n_nodes:', n_nodes)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        feature = graph.ndata.pop(
            'feats'
        ).to('cuda')
        n_features = feature.size(-1)
        print('n_features:', n_features)

        # inputs
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
        with utils.Profiler(n_epochs) as prof:
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
        n_nodes = 0
        for _, _, dty in graph.canonical_etypes:
            n_nodes += graph.num_dst_nodes(dty)
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
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = model(graph)[category]
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_nodes / timing))

    @staticmethod
    def _bench_pyg_homo(dataset, model, d_hidden):
        # info
        print('[PYG] {}, {}, d_hidden={}'.format(
            type(dataset).__name__,
            model.__name__, d_hidden
        ))

        # dataset
        n_epochs = 20
        graph = dataset[0].to('cuda')
        n_nodes = graph.num_nodes
        print('n_nodes:', n_nodes)
        n_labels = dataset.num_classes
        print('n_labels:', n_labels)
        n_features = graph.num_features
        print('n_features:', n_features)

        # inputs
        gradient = torch.ones(
            [n_nodes, n_labels]
        ).to('cuda')
        model = model(
            in_features=n_features,
            gnn_features=d_hidden,
            out_features=n_labels
        ).to('cuda')

        # prewarm
        y = model(graph)
        y.backward(gradient=gradient)
        torch.cuda.synchronize()

        # training
        timing = None
        time.sleep(2.0)
        print('[TRAINING]')
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = model(graph)
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
        print('n_nodes:', n_nodes)
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
        with utils.Profiler(n_epochs) as prof:
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
        n_nodes = 0
        for _, _, dty in dglgraph.canonical_etypes:
            n_nodes += dglgraph.num_dst_nodes(dty)
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
        stitcher = sageir.Stitcher()
        dataflow = mod2ir.transform(
            model, kwargs=kwargs
        )[category]
        dataflow = optimizer.lower(
            dataflow, kwargs=kwargs
        )
        dataflow = stitcher.transform(
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
        with utils.Profiler(n_epochs) as prof:
            for _ in range(n_epochs):
                y = executor.run(
                    dataflow, kwargs=kwargs
                )
                y.backward(gradient=gradient)
            torch.cuda.synchronize()
            timing = prof.timing() / n_epochs
        print('throughput: {:.1f}'.format(n_nodes / timing))


class Benchmark(BenchMethods):
    HOM_MODELS = [
        (DGLGATModel, GATModel),
        (DGLGCNModel, GCNModel)
    ]
    HET_MODELS = [
        (DGLRGATModel, RGATModel)
    ]
    HOM_DATASETS = [
        'cora_tiny', 'amazon', 'cora_full', 'reddit',
    ]
    HET_DATASETS = [
        'aifb_hetero', 'mutag_hetero', 'bgs_hetero', 'am_hetero'
    ]
    DGL_DATASETS = {
        'cora_tiny': cit.CoraGraphDataset,  # 2.7k
        'amazon': bench.AmazonCoBuyPhotoDataset,  # 7.7k
        'cora_full': bench.CoraFullDataset,  # 19.8k
        'reddit': reddit.RedditDataset,  # 233.0k
        #
        'aifb_hetero': rdf.AIFBDataset,  # 7.3k
        'mutag_hetero': rdf.MUTAGDataset,  # 27.2k
        'bgs_hetero': rdf.BGSDataset,  # 94.8k
        'am_hetero': rdf.AMDataset  # 881.7k
    }
    PYG_DATASETS = {
        'cora_tiny': lambda: pygds.Planetoid(root='.data', name='Cora'),
        'amazon': lambda: pygds.Amazon(root='.data', name='Photo'),
        'cora_full': lambda: pygds.CoraFull(root='.data'),
        'reddit': lambda: pygds.Reddit(root='.data'),
        #
        'aifb_hetero': lambda: pygds.Reddit(root='.data', name='AIFB'),
        'mutag_hetero': lambda: pygds.Reddit(root='.data', name='MUTAG'),
        'bgs_hetero': lambda: pygds.Reddit(root='.data', name='BGS'),
        'am_hetero': lambda: pygds.Reddit(root='.data', name='AM'),
    }

    def dataset_info(self):
        for dataset in self.HOM_DATASETS:
            dataset = self.DGL_DATASETS[
                dataset
            ](verbose=False)
            self._check_dataset(dataset)
        for dataset in self.HET_DATASETS:
            dataset = self.DGL_DATASETS[
                dataset
            ](verbose=False)
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
    benchmark = Benchmark()
    benchmark.dataset_info()
    # bench.bench_heterogenous()


if __name__ == "__main__":
    main()
