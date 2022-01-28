import torch
import sageir
from sageir import mp
from torch import nn, optim
from dgl.data.rdf import MUTAGDataset
from common.model import GATLayer, HeteroGraphConv
from tqdm import tqdm


def dump_data():
    dataset = MUTAGDataset(
        verbose=True
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)
    pred_cat = dataset.predict_category
    label = dglgraph.ndata.pop(
        'labels'
    )[pred_cat].type(
        torch.LongTensor
    ).to('cuda')
    n_labels = dataset.num_classes
    test_mask = dglgraph.ndata.pop(
        'test_mask'
    )[pred_cat].type(
        torch.BoolTensor
    ).to('cuda')
    train_mask = dglgraph.ndata.pop(
        'train_mask'
    )[pred_cat].type(
        torch.BoolTensor
    ).to('cuda')
    torch.save(
        {
            'graph': graph,
            'label': label,
            'pred_cat': pred_cat,
            'n_labels': n_labels,
            'test_mask': test_mask,
            'train_mask': train_mask,
        },
        '.data/{}.pt'.format(dataset.name)
    )


def check_hetero():
    dataset = torch.load(
        '.data/mutag-hetero.pt'
    )
    graph = dataset['graph']
    label = dataset['label']
    pred_cat = dataset['pred_cat']
    n_labels = dataset['n_labels']
    test_mask = dataset['test_mask']
    train_mask = dataset['train_mask']
    print('predict_category:', pred_cat)

    #
    n_heads = 2
    d_hidden = 16
    model = HeteroGraphConv({
        ety: GATLayer(
            in_features=d_hidden,
            out_features=d_hidden,
            n_heads=n_heads
        )
        for ety in graph.etypes
    }).to('cuda')
    features = {
        nty: torch.randn(size=[
            graph.nty2num[nty],
            d_hidden
        ]).to('cuda')
        for nty in graph.nty2num
    }

    #
    print('===== mod2ir =====')
    mod2ir = sageir.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs={
            'hgraph': graph,
            'xs': features
        }
    )
    sageir.Printer().dump(dataflow)

    #
    print('===== optimizer =====')
    optimizer = sageir.Optimizer()
    dataflow = optimizer.lower(dataflow)
    sageir.Printer().dump(dataflow)

    #
    print('===== executor =====')
    executor = sageir.Executor()
    logits = executor.run(
        dataflow[pred_cat],
        kwargs={
            'hgraph': graph,
            'xs': features
        }
    )

    a = 0


def test():
    for _ in tqdm(range(10)):
        check_hetero()


if __name__ == "__main__":
    test()
