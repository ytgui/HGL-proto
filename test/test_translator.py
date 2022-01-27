import torch
import sageir
from sageir import mp
from torch import nn, optim
from dgl.data import CoraGraphDataset, RedditDataset
from tqdm import tqdm


class GATLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int):
        super().__init__()
        #
        self.n_heads = n_heads
        self.n_features = out_features
        self.linear_q = nn.Linear(n_heads * out_features, n_heads)
        self.linear_k = nn.Linear(n_heads * out_features, n_heads)
        self.linear_v = nn.Linear(in_features, n_heads * out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph: mp.Graph, x: torch.Tensor):
        # different to bert
        h = self.linear_v(x)
        q = self.linear_q(h)
        k = self.linear_k(h)
        h = h.view(size=[
            -1, self.n_heads,
            self.n_features
        ])
        graph.src_node['u'] = h
        graph.src_node['q'] = q
        graph.dst_node['k'] = k

        # gat attention
        graph.message_func(mp.Fn.u_add_v('q', 'k', 'e'))
        graph.edge['coeff'] = self.leaky_relu(graph.edge['e'])
        graph.message_func(mp.Fn.edge_softmax('coeff', 'attn'))
        graph.message_func(mp.Fn.u_mul_e('u', 'attn', 'm'))
        graph.reduce_func(mp.Fn.aggregate_sum('m', 'v'))
        return torch.mean(graph.dst_node['v'], dim=1)


class GATModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int):
        nn.Module.__init__(self)
        #
        self.i2h = GATLayer(
            in_features, gnn_features, n_heads=n_heads
        )
        self.h2o = GATLayer(
            gnn_features, out_features, n_heads=n_heads
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self, graph: mp.Graph, x: torch.Tensor):
        h = self.i2h(graph, x)
        h = self.activation(h)
        h = self.h2o(graph, h)
        return h


def check_homo():
    dataset = CoraGraphDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)

    #
    labels = dglgraph.ndata.pop(
        'label'
    ).type(torch.LongTensor)
    feature = dglgraph.ndata.pop(
        'feat'
    ).type(torch.FloatTensor)
    test_mask = dglgraph.ndata.pop(
        'test_mask'
    ).type(torch.BoolTensor)
    train_mask = dglgraph.ndata.pop(
        'train_mask'
    ).type(torch.BoolTensor)
    n_nodes = dglgraph.num_nodes()
    n_labels = dataset.num_classes
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
    feature = feature.to('cuda')
    labels = labels.to('cuda')

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
    executor = sageir.Executor()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3, weight_decay=1e-5
    )

    def evaluate(features, labels):
        model.eval()
        with torch.no_grad():
            logits = executor.run(
                dataflow, kwargs={
                    'graph': graph,
                    'x': features
                }
            )
            assert logits.dim() == 2
            logits = logits[test_mask]
            labels = labels[test_mask]
            indices = torch.argmax(logits, dim=-1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    for epoch in range(10):
        for _ in range(20):
            model.train()
            logits = executor.run(
                dataflow, kwargs={
                    'graph': graph,
                    'x': feature
                }
            )
            loss = loss_fn(
                logits[train_mask],
                target=labels[train_mask]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        accuracy = evaluate(feature, labels)
        print('[epoch={}] loss: {:.04f}, accuracy: {:.02f}'.format(
            epoch, loss.item(), accuracy
        ))
        if accuracy > 0.75:
            break
    else:
        raise RuntimeError("not convergent")


def test():
    for _ in tqdm(range(10)):
        check_homo()


if __name__ == "__main__":
    test()
