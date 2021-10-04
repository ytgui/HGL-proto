import time
import random
import torch
from torch import nn


class Sampler:
    def __init__(self,
                 rules: list):
        self.rules = rules

    def collect_fn(self,
                   adjacency: torch.Tensor,
                   dst_nodes: torch.Tensor,
                   device: str):
        blocks = []
        dst_nodes = sorted(
            set(dst_nodes.tolist())
        )
        frontier = None
        starting = None
        ending = torch.LongTensor(dst_nodes)
        for n in self.rules:
            # dst nodes
            dst2src = {}
            for node in dst_nodes:
                neighbors = torch.nonzero(
                    adjacency[node]
                ).squeeze(-1).tolist()
                random.shuffle(neighbors)
                dst2src[node] = neighbors[:n]
            # src nodes
            src_nodes = set()
            for dst in dst2src:
                for src in dst2src[dst]:
                    src_nodes.add(src)
            src_nodes = sorted(list(src_nodes))
            # build index
            node2idx = {
                node: i
                for i, node in enumerate(dst_nodes)
            }
            for node in src_nodes:
                if node in node2idx:
                    continue
                node2idx[node] = len(node2idx)
            # build block
            block = torch.zeros([
                len(dst_nodes), len(node2idx)
            ])
            for dst in dst2src:
                n_src = len(dst2src[dst])
                for src in dst2src[dst]:
                    block[
                        node2idx[dst], node2idx[src]
                    ] = 1.0 / n_src
            blocks.append(block.to(device))
            # next hop
            frontier = sorted(
                node2idx.keys(),
                key=lambda node: node2idx[node]
            )
            dst_nodes = frontier
        #
        starting = torch.LongTensor(frontier)
        return starting, ending, list(reversed(blocks))


class DataLoader:
    def __init__(self,
                 adjacency: torch.Tensor,
                 candidates: torch.Tensor,
                 sampler: Sampler,
                 batch_size: int,
                 device: str):
        self.adjacency = adjacency
        self.candidates = candidates
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        n_nodes = len(self.candidates)
        indices = torch.randperm(n_nodes)
        self.candidates = self.candidates[indices]
        for left in range(0, n_nodes, self.batch_size):
            right = left + self.batch_size
            nodes = self.candidates[left:right]
            yield self.sampler.collect_fn(
                self.adjacency,
                dst_nodes=nodes,
                device=self.device
            )


class SageConv(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.fc_self = nn.Linear(in_features, out_features, bias=False)
        self.fc_neighbor = nn.Linear(in_features, out_features, bias=False)

    def forward(self, block, x):
        n_dst = block.size(0)
        h_self = self.fc_self(
            x[:n_dst]
        )
        h_neighbor = torch.matmul(
            block,
            self.fc_neighbor(x)
        )
        return h_self + h_neighbor


class SageDense(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float):
        nn.Module.__init__(self)
        self.layer1 = SageConv(in_features, hidden_features)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer2 = SageConv(hidden_features, out_features)

    def forward(self, blocks, x):
        h = self.layer1(blocks[0], x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.layer2(blocks[1], h)
        return h


def check_sage_training(dataset):
    #
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('device: {}'.format(device))

    #
    labels = dataset['labels']
    features = dataset['features']
    adjacency = dataset['adjacency']
    val_mask = dataset['val_mask']
    test_mask = dataset['test_mask']
    train_mask = dataset['train_mask']
    val_nodes = torch.nonzero(val_mask).squeeze()
    test_nodes = torch.nonzero(test_mask).squeeze()
    train_nodes = torch.nonzero(train_mask).squeeze()
    print('dataset split: {}-{}-{}'.format(
        len(train_nodes), len(val_nodes), len(test_nodes)))

    #
    batch_size = 16
    sampler_neighboors = 8
    n_features = features.size(1)
    n_labels = torch.max(labels) + 1
    model = SageDense(
        in_features=n_features,
        hidden_features=32,
        out_features=n_labels,
        dropout=0.1
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)

    #
    def train(src, dst, blocks):
        model.train()
        logits = model(blocks, features[src].to(device))
        loss = loss_fn(logits, labels[dst].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(eval_nodes):
        model.eval()
        accuracy = []
        dataloader = DataLoader(
            adjacency=adjacency,
            candidates=eval_nodes,
            sampler=Sampler([sampler_neighboors] * 2),
            batch_size=batch_size,
            device=device
        )
        for i, (src, dst, blocks) in enumerate(dataloader):
            if i == 100:
                break
            with torch.no_grad():
                logits = model(blocks, features[src].to(device))
                predict = torch.argmax(logits, dim=-1)
                correct = torch.sum(predict == labels[dst].to(device))
                accuracy.append(correct.item() / len(dst))
        return sum(accuracy) / len(accuracy)

    #
    dataloader = DataLoader(
        adjacency=adjacency,
        candidates=train_nodes,
        sampler=Sampler([sampler_neighboors] * 2),
        batch_size=batch_size,
        device=device
    )
    best_accuracy = 0.0
    for epoch in range(20):
        for i, (src, dst, blocks) in enumerate(dataloader):
            loss = train(src, dst, blocks)
            val_accuracy = evaluate(val_nodes)
            print('[{}-{}] loss: {:.3f}, val_accuracy: {:.3f}'.format(
                epoch, i, loss, val_accuracy))
        test_accuracy = evaluate(test_nodes)
        best_accuracy = max(best_accuracy, test_accuracy)
        print('[{}] test_accuracy: {:.3f}, best_accuracy: {:.3f}'.format(
            epoch, test_accuracy, best_accuracy))
        # if torch.cuda.is_available():
        #     print(torch.cuda.memory_summary())
        if test_accuracy >= 0.80:
            break


def test():
    # dataset
    dataset = torch.load('.data/cora_v2.pt')

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
