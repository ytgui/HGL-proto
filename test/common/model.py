import torch
from torch import nn
from sageir import mp
from typing import Union


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

    def forward(self, graph: mp.Graph, x: Union[torch.Tensor, list]):
        # different to bert
        if isinstance(x, (list, tuple)):
            assert len(x) == 2
            src_size = x[0].size(0)
            dst_size = x[1].size(0)
            graph.blk.size[0] == src_size
            graph.blk.size[1] == dst_size
            #
            h_src = self.linear_v(x[0])
            h_dst = self.linear_v(x[1])
        elif isinstance(x, torch.Tensor):
            h_src = h_dst = self.linear_v(x)
        else:
            raise TypeError
        q = self.linear_q(h_dst)
        k = self.linear_k(h_src)
        h_src = h_src.view(size=[
            -1, self.n_heads,
            self.n_features
        ])
        graph.src_node['q'] = q
        graph.dst_node['k'] = k
        graph.src_node['u'] = h_src

        # gat attention
        graph.message_func(mp.Fn.u_add_v('k', 'q', 'e'))
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


class HeteroGraphConv(nn.Module):
    def __init__(self, convs: dict):
        nn.Module.__init__(self)
        self._convs = nn.ModuleDict(convs)

    def forward(self, hgraph, xs):
        counts = {}
        outputs = {}
        for (sty, ety, dty), graph \
                in hgraph:
            res = self._convs[ety](
                graph, (xs[sty], xs[dty])
            )
            if dty not in counts:
                counts[dty] = 1
            else:
                counts[dty] += 1
            if dty not in outputs:
                outputs[dty] = res
            else:
                exist = outputs[dty]
                outputs[dty] = exist + res
        for dty, res in outputs.items():
            outputs[dty] = res / counts[dty]
        return outputs
