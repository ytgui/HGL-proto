import torch
import sageir
from torch import nn
from sageir import mp
from dgl.data import CoraGraphDataset, RedditDataset
from tqdm import tqdm


class GATLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 n_heads: int,
                 out_features: int):
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
        return graph.dst_node['v']


def check_homo():
    dataset = CoraGraphDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)

    #
    label = dglgraph.ndata.pop(
        'label'
    ).type(torch.IntTensor)
    feature = dglgraph.ndata.pop(
        'feat'
    ).type(torch.FloatTensor)
    n_nodes = dglgraph.num_nodes()
    n_labels = dataset.num_classes
    n_features = feature.size(1)

    #
    n_heads = 8
    d_hidden = 16
    model = GATLayer(
        in_features=n_features,
        n_heads=n_heads,
        out_features=d_hidden
    ).to('cuda')
    feature = feature.to('cuda')

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
    y = executor.run(
        dataflow, kwargs={
            'graph': graph,
            'x': feature
        }
    )

    a = 0


def test():
    for _ in tqdm(range(10)):
        check_homo()


if __name__ == "__main__":
    test()
