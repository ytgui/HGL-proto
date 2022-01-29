import torch
import sageir
from sageir import mp
from torch import nn, optim
from dgl.data.rdf import MUTAGDataset
from common.model import RGATModel
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
    model = RGATModel(
        hgraph=graph,
        in_features=d_hidden,
        gnn_features=d_hidden,
        out_features=d_hidden,
        n_heads=n_heads
    ).to('cuda')
    node_indices = {
        nty: torch.linspace(
            0, num - 1, num,
            dtype=torch.int64
        ).to('cuda')
        for nty, num in graph.nty2num.items()
    }

    #
    print('===== mod2ir =====')
    mod2ir = sageir.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs={
            'hgraph': graph,
            'xs': node_indices
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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3, weight_decay=1e-5
    )

    #
    def train():
        executor.train()
        logits = executor.run(
            dataflow[pred_cat],
            kwargs={
                'hgraph': graph,
                'xs': node_indices
            }
        )
        loss = loss_fn(
            logits[train_mask],
            target=label[train_mask]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate():
        executor.eval()
        with torch.no_grad():
            logits = executor.run(
                dataflow[pred_cat],
                kwargs={
                    'hgraph': graph,
                    'xs': node_indices
                }
            )
            assert logits.dim() == 2
            indices = torch.argmax(
                logits[test_mask], dim=-1
            )
            correct = torch.sum(
                indices == label[test_mask]
            )
            return correct.item() / len(indices)

    #
    for epoch in range(20):
        loss_val = None
        for _ in range(10):
            loss_val = train()
        #
        accuracy = evaluate()
        print('[epoch={}] loss: {:.04f}, accuracy: {:.02f}'.format(
            epoch, loss_val, accuracy
        ))
        if accuracy > 0.6:
            break
    else:
        raise RuntimeError("not convergent")


def test():
    for _ in tqdm(range(10)):
        check_hetero()


if __name__ == "__main__":
    test()
