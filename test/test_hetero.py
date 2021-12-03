import torch
from torch import fx, nn
import dgl
from dgl import nn as dglnn
from typing import Dict


class HeteroGraph:
    def __init__(self,
                 hetero_dict: Dict[tuple, tuple]):
        # parse hetero dict
        self._node2num = {}
        for (sty, _, dty), data in hetero_dict.items():
            if sty not in self._node2num:
                self._node2num[sty] = 0
            self._node2num[sty] = max(
                self._node2num[sty],
                max([src for src, _ in data]) + 1
            )
            if dty not in self._node2num:
                self._node2num[dty] = 0
            self._node2num[dty] = max(
                self._node2num[dty],
                max([dst for _, dst in data]) + 1
            )
        # generate hetero graph
        self._hetero_graph = {}
        for (sty, ety, dty), data in hetero_dict.items():
            if (sty, ety, dty) not in self._hetero_graph:
                self._hetero_graph[
                    sty, ety, dty
                ] = torch.zeros([
                    self._node2num[dty],
                    self._node2num[sty],
                ])
            for src, dst in data:
                self._hetero_graph[
                    sty, ety, dty
                ][dst, src] = 1.0
        # generate extra index
        self._src2index = {}

    def __iter__(self):
        return iter(self._hetero_graph.items())


class DenseSAGEConv(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.fc_self = nn.Linear(
            in_features, out_features, bias=False
        )
        self.fc_neighbor = nn.Linear(
            in_features, out_features, bias=False
        )

    def forward(self, block, src, dst):
        h_self = self.fc_self(
            dst
        )
        h_neighbor = torch.matmul(
            block,
            self.fc_neighbor(src)
        )
        return h_self + h_neighbor


class HeteroGraphConv(nn.Module):
    def __init__(self, hetero_convs: Dict[str, nn.Module]):
        nn.Module.__init__(self)
        self._convs = hetero_convs

    def forward(self, hetero_block, features):
        outputs = {}
        for (sty, ety, dty), block in hetero_block:
            src = features[sty]
            dst = features[dty]
            layer = self._convs[ety]
            if dty not in outputs:
                outputs[dty] = []
            outputs[dty].append(
                layer(block, src, dst)
            )
        for dty, items in outputs.items():
            outputs[dty] = torch.mean(
                torch.stack(items), dim=0
            )
        return outputs


def test_hetero():
    hetero_dict = {
        ('user', 'plays', 'video'): [
            [1, 0], [1, 1]
        ],
        ('user', 'follows', 'user'): [
            [0, 1], [1, 2]
        ],
    }
    features = {
        'user': torch.FloatTensor([
            [0.1], [0.3], [0.5]
        ]),
        'video': torch.FloatTensor([
            [-1.0], [1.0]
        ]),
    }

    # dgl
    dgl_graph = dgl.heterograph(hetero_dict)
    dgl_conv = dglnn.HeteroGraphConv(
        {
            'plays': dglnn.SAGEConv(
                1, 1, aggregator_type='mean'
            ),
            'follows': dglnn.SAGEConv(
                1, 1, aggregator_type='mean'
            ),
        },
        aggregate='mean'
    )
    dgl_out = dgl_conv(dgl_graph, features)

    # hetero
    dense_graph = HeteroGraph(
        hetero_dict=hetero_dict
    )
    dense_conv = HeteroGraphConv(
        hetero_convs={
            'plays': DenseSAGEConv(1, 1),
            'follows': DenseSAGEConv(1, 1)
        }
    )
    with torch.no_grad():
        param = dense_conv._convs
        param['plays'].fc_self.weight.copy_(
            dgl_conv.mods['plays'].fc_self.weight
        )
        param['plays'].fc_neighbor.weight.copy_(
            dgl_conv.mods['plays'].fc_neigh.weight
        )
        param['follows'].fc_self.weight.copy_(
            dgl_conv.mods['follows'].fc_self.weight
        )
        param['follows'].fc_neighbor.weight.copy_(
            dgl_conv.mods['follows'].fc_neigh.weight
        )
    dense_out = dense_conv(dense_graph, features)

    # check
    for nty in features.keys():
        assert torch.allclose(
            dgl_out[nty], dense_out[nty], atol=1e-5, rtol=1e-3
        )

    #
    a = 0


def test():
    test_hetero()


if __name__ == "__main__":
    # 1. hetero graph
    # 2. hetero gcn, gat
    test()
