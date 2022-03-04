import torch
import sageir
from sageir import mp
from torch import nn, optim
from dgl.data.rdf import AIFBDataset
from common.model import RGATModel


def check_stitch():
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
    mod2ir = sageir.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs=kwargs
    )[pred_cat]
    sageir.Printer().dump(dataflow)

    #
    print('===== lower =====')
    optimizer = sageir.Optimizer()
    dataflow_1 = optimizer.lower(
        dataflow, kwargs=kwargs
    )
    sageir.Printer().dump(dataflow)

    #
    print('===== stitch =====')
    stitcher = sageir.Stitcher()
    dataflow_2 = stitcher.transform(
        dataflow_1, kwargs=kwargs
    )
    sageir.Printer().dump(dataflow)

    #
    print('===== executor =====')
    executor = sageir.Executor()
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
    check_stitch()


if __name__ == "__main__":
    test()
