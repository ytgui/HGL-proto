import time
import torch
from torch import nn
from sageir import mp, sparse, bundle
from dgl.data import CoraFullDataset


def bench_gspmm():
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


def bench_bundle():
    n_nodes = 64
    n_features = 64
    x = torch.randn(
        [n_nodes, n_features]
    ).to('cuda')
    x.requires_grad = True

    #
    d_hidden = 16
    fc_1 = nn.Linear(
        n_features, d_hidden, bias=False
    ).to('cuda')
    fc_2 = nn.Linear(
        n_features, d_hidden, bias=False
    ).to('cuda')
    torch.cuda.synchronize()

    #
    origin = []
    bundled = []
    for _ in range(100):
        before = time.time()
        y_1, y_2 = fc_1(x), fc_2(x)
        torch.sum(y_1 + y_2).backward()
        torch.cuda.synchronize()
        origin.append(time.time() - before)
        #
        before = time.time()
        y_3, y_4 = bundle.GEMMBundleFunction.apply(
            x, fc_1.weight, fc_2.weight,
            fc_1.bias, fc_2.bias
        )
        torch.sum(y_3 + y_4).backward()
        torch.cuda.synchronize()
        bundled.append(time.time() - before)
    origin = sorted(origin)[5:-5]
    bundled = sorted(bundled)[5:-5]
    print('origin: {:.3f}, bundled: {:.3f}'.format(
        sum(origin), sum(bundled)
    ))


def main():
    bench_bundle()


if __name__ == "__main__":
    main()
