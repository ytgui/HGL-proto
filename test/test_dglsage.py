import time
from dgl import data
import torch
from torch import nn
import dgl
import dgl.nn as dgl_nn
import dgl.data as dgl_data
import dgl.dataloading as dgl_loader


class DGLSage(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float):
        nn.Module.__init__(self)
        self.layer1 = dgl_nn.SAGEConv(
            in_features, hidden_features, 'mean'
        )
        self.activation = nn.ReLU(
            inplace=True
        )
        self.dropout = nn.Dropout(
            p=dropout
        )
        self.layer2 = dgl_nn.SAGEConv(
            hidden_features, out_features, 'mean'
        )

    def forward(self, blocks, x):
        h = self.layer1(blocks[0], x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.layer2(blocks[1], h)
        return h


def check_sage_training(dataset):
    graph = dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    train_mask = graph.ndata['train_mask']
    val_nodes = torch.nonzero(val_mask).squeeze()
    test_nodes = torch.nonzero(test_mask).squeeze()
    train_nodes = torch.nonzero(train_mask).squeeze()

    #
    n_labels = dataset.num_classes
    _, n_features = features.size()
    model = DGLSage(
        in_features=n_features,
        hidden_features=32,
        out_features=n_labels,
        dropout=0.1
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)

    #
    def train(src, dst, blocks):
        model.train()
        logits = model(blocks, features[src])
        loss = loss_fn(logits, labels[dst])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(eval_nodes):
        model.eval()
        dataloader = dgl_loader.NodeDataLoader(
            graph,
            eval_nodes,
            dgl_loader.MultiLayerFullNeighborSampler(2),
            batch_size=len(eval_nodes),
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        dataiter = iter(dataloader)
        src, dst, blocks = next(dataiter)
        with torch.no_grad():
            logits = model(blocks, features[src])
            predict = torch.argmax(logits, dim=-1)
            correct = torch.sum(predict == labels[dst])
            return correct.item() / len(dst)

    #
    sampler = dgl_loader.MultiLayerNeighborSampler(
        [4, 4]
    )
    dataloader = dgl_loader.NodeDataLoader(
        graph, train_nodes, sampler,
        batch_size=64, shuffle=True,
        drop_last=False, num_workers=0
    )
    best_accuracy = 0.0
    for epoch in range(20):
        for i, (src, dst, blocks) in enumerate(dataloader):
            print('[{}-{}] blocks: {}-{}-{}'.format(
                epoch, i,
                blocks[0].num_src_nodes(),
                blocks[0].num_dst_nodes(),
                blocks[1].num_dst_nodes()))
            loss = train(src, dst, blocks)
            val_accuracy = evaluate(val_nodes)
            print('[{}-{}] loss: {:.3f}, val_accuracy: {:.3f}'.format(
                epoch, i, loss, val_accuracy))
        test_accuracy = evaluate(test_nodes)
        best_accuracy = max(best_accuracy, test_accuracy)
        print('[{}] test_accuracy: {}, best_accuracy: {}'.format(
            epoch, test_accuracy, best_accuracy))
        if test_accuracy >= 0.85:
            break


def test():
    # dataset
    dataset = dgl_data.CoraGraphDataset(
        verbose=False
    )

    # profile
    import pstats
    import cProfile
    profiler = cProfile.Profile()

    #
    profiler.enable()
    check_sage_training(dataset)
    profiler.disable()
    pstats.Stats(profiler).strip_dirs() \
        .sort_stats(pstats.SortKey.TIME).print_stats(20)


if __name__ == "__main__":
    test()
