import torch
import dgl
from dgl import nn as dglnn


def test_dgl_hetero():
    data_dict = {
        ('user', 'plays', 'video'): [
            torch.tensor([1, 0]),
            torch.tensor([1, 1])
        ],
        ('user', 'follows', 'user'): [
            torch.tensor([0, 1]),
            torch.tensor([1, 2])
        ],
    }
    graph = dgl.heterograph(data_dict)
    features = {
        'user': torch.FloatTensor([[0.1], [0.3], [0.5]]),
        'video': torch.FloatTensor([[-1.0], [1.0]]),
    }

    #
    conv = dglnn.HeteroGraphConv(
        {
            'plays': dglnn.GraphConv(1, 1),
            'follows': dglnn.GraphConv(1, 1),
        },
        aggregate='sum'
    )
    y_out = conv(graph, features)
    print(y_out['user'].flatten())
    print(y_out['video'].flatten())
    a = 0


def test():
    test_dgl_hetero()


if __name__ == "__main__":
    # 1. hetero graph
    # 2. hetero gcn, gat
    # 3. ? rel_conv vs hetero_conv
    test()
