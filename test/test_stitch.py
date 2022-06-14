import torch
import hgl
from hgl import mp
from dgl.data.rdf import AIFBDataset
from common.model import RGCNModel, RGATModel


def check_stitch_gcn():
    #
    dataset = AIFBDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)
    pred_cat = dataset.predict_category
    print('predict_category:', pred_cat)

    #
    n_labels = 4
    d_hidden = 16
    model = RGCNModel(
        hgraph=graph,
        in_features=d_hidden,
        gnn_features=d_hidden,
        out_features=n_labels
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
        'xs': node_indices,
        'norms': {
            str(rel): g.right_norm()
            for rel, g in graph.hetero_graph.items()
        }
    })

    #
    print('===== mod2ir =====')
    mod2ir = hgl.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs=kwargs
    )[pred_cat]
    hgl.Printer().dump(dataflow)

    #
    print('===== lower =====')
    optimizer = hgl.Optimizer()
    dataflow_1 = optimizer.lower(
        dataflow, kwargs=kwargs
    )
    hgl.Printer().dump(dataflow)

    #
    print('===== stitch =====')
    stitcher = hgl.Stitcher()
    dataflow_2 = stitcher.transform(
        dataflow_1, kwargs=kwargs
    )
    hgl.Printer().dump(dataflow)

    #
    print('===== executor =====')
    executor = hgl.Executor()
    executor.eval()
    y_1 = executor.run(
        dataflow_1,
        kwargs=kwargs
    )
    y_2 = executor.run(
        dataflow_2,
        kwargs=kwargs
    )
    assert torch.allclose(y_1, y_2, atol=1e-3)


def check_stitch_gat():
    #
    dataset = AIFBDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)
    pred_cat = dataset.predict_category
    print('predict_category:', pred_cat)

    #
    n_heads = 2
    n_labels = 4
    d_hidden = 16
    model = RGATModel(
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

    #
    print('===== mod2ir =====')
    mod2ir = hgl.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs=kwargs
    )[pred_cat]
    hgl.Printer().dump(dataflow)

    #
    print('===== lower =====')
    optimizer = hgl.Optimizer()
    dataflow_1 = optimizer.lower(
        dataflow, kwargs=kwargs
    )
    hgl.Printer().dump(dataflow)

    #
    print('===== stitch =====')
    stitcher = hgl.Stitcher()
    dataflow_2 = stitcher.transform(
        dataflow_1, kwargs=kwargs
    )
    hgl.Printer().dump(dataflow)

    #
    print('===== executor =====')
    executor = hgl.Executor()
    executor.eval()
    y_1 = executor.run(
        dataflow_1,
        kwargs=kwargs
    )
    y_2 = executor.run(
        dataflow_2,
        kwargs=kwargs
    )
    assert torch.allclose(y_1, y_2, atol=1e-3)


def test():
    check_stitch_gcn()
    check_stitch_gat()


if __name__ == "__main__":
    test()
