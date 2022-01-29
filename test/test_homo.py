import torch
import sageir
from sageir import mp
from torch import nn, optim
from dgl.data import CoraGraphDataset
from common.model import GATModel
from tqdm import tqdm


def dump_data():
    dataset = CoraGraphDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)
    label = dglgraph.ndata.pop(
        'label'
    ).type(torch.LongTensor).to('cuda')
    feature = dglgraph.ndata.pop(
        'feat'
    ).type(torch.FloatTensor).to('cuda')
    test_mask = dglgraph.ndata.pop(
        'test_mask'
    ).type(torch.BoolTensor).to('cuda')
    train_mask = dglgraph.ndata.pop(
        'train_mask'
    ).type(torch.BoolTensor).to('cuda')
    n_labels = dataset.num_classes
    torch.save(
        {
            'graph': graph,
            'label': label,
            'feature': feature,
            'n_labels': n_labels,
            'test_mask': test_mask,
            'train_mask': train_mask
        },
        '.data/{}.pt'.format(dataset.name)
    )


def check_homo():
    dataset = torch.load(
        '.data/cora_v2.pt'
    )
    graph = dataset['graph']
    label = dataset['label']
    feature = dataset['feature']
    n_labels = dataset['n_labels']
    test_mask = dataset['test_mask']
    train_mask = dataset['train_mask']
    n_features = feature.size(1)

    #
    n_heads = 8
    d_hidden = 32
    model = GATModel(
        in_features=n_features,
        gnn_features=d_hidden,
        out_features=n_labels,
        n_heads=n_heads,
    ).to('cuda')

    #
    print('===== mod2ir =====')
    mod2ir = sageir.Module2IR()
    dataflow = mod2ir.transform(
        model, kwargs={
            'graph': graph,
            'x': feature
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
        lr=2e-4, weight_decay=1e-5
    )

    #
    def train():
        executor.train()
        logits = executor.run(
            dataflow, kwargs={
                'graph': graph,
                'x': feature
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
                dataflow, kwargs={
                    'graph': graph,
                    'x': feature
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
        for _ in range(50):
            loss_val = train()
        #
        accuracy = evaluate()
        print('[epoch={}] loss: {:.04f}, accuracy: {:.02f}'.format(
            epoch, loss_val, accuracy
        ))
        if accuracy > 0.72:
            break
    else:
        raise RuntimeError("not convergent")


def test():
    for _ in tqdm(range(10)):
        check_homo()


if __name__ == "__main__":
    test()
