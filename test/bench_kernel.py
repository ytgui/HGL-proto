import time
import torch
from sageir import mp, sparse
from dgl.data import CoraFullDataset


def main():
    dataset = CoraFullDataset(
        verbose=False
    )
    dglgraph = dataset[0].to('cuda')
    graph = mp.from_dglgraph(dglgraph)
    n_nodes = dglgraph.num_nodes()

    # inputs
    n_features = 16
    feature = torch.randn(
        size=[n_nodes, 1, n_features],
        requires_grad=True
    ).to('cuda')
    gradient = torch.ones_like(feature)
    torch.cuda.synchronize()

    #
    forward = []
    backward = []
    for _ in range(100):
        before = time.time()
        y = sparse.gspmm(
            block=graph.blk,
            edge=None, x=feature
        )
        torch.cuda.synchronize()
        forward.append(time.time() - before)
        #
        before = time.time()
        y.backward(gradient=gradient)
        torch.cuda.synchronize()
        backward.append(time.time() - before)
    forward = sorted(forward)[5:-5]
    backward = sorted(backward)[5:-5]
    print('forward: {:.3f}, backward: {:.3f}'.format(
        sum(forward), sum(backward)
    ))


if __name__ == "__main__":
    main()
