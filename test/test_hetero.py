import torch
from torch import fx, nn
from typing import Dict


class HeteroGraph:
    def __init__(self,
                 hetero_dict: Dict[tuple, torch.Tensor]):
        # parse hetero dict
        self._node2num = {}
        for (sty, _, dty), data in hetero_dict.items():
            assert data.dim() == 2
            assert data.size(1) == 2
            if sty not in self._node2num:
                self._node2num[sty] = 0
            self._node2num[sty] = max(
                self._node2num[sty],
                data[:, 0].max().item() + 1
            )
            if dty not in self._node2num:
                self._node2num[dty] = 0
            self._node2num[dty] = max(
                self._node2num[dty],
                data[:, 1].max().item() + 1
            )
        # generate hetero graph
        a = 0


def test_hetero():
    hetero_dict = {
        (
            'user', 'plays', 'video'
        ): torch.tensor([
            [1, 0], [1, 1]
        ]),
        (
            'user', 'follows', 'user'
        ): torch.tensor([
            [0, 1], [1, 2]
        ]),
    }
    graph = HeteroGraph(hetero_dict)
    features = {
        'user': torch.tensor([
            [0.1], [0.3], [0.5]
        ]),
        'video': torch.tensor([
            [-1.0], [1.0]
        ]),
    }

    #

    #
    # dag = fx.symbolic_trace(conv)
    a = 0


def test():
    test_hetero()


if __name__ == "__main__":
    # 1. hetero graph
    # 2. hetero gcn, gat
    test()
