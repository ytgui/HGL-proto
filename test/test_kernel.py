import torch
import random
import sageir
from torch import nn
from tqdm import tqdm
from dgl import nn as dglnn
from dgl.data import CoraGraphDataset


def to_dense(n_rows, n_cols, sparse):
    indptr, indices = sparse
    dense = torch.zeros(size=[n_rows, n_cols])
    for row in range(n_rows):
        for i in range(indptr[row], indptr[row + 1]):
            col = indices[i]
            dense[row, col] = 1.0
    return dense


def to_csr(dense):
    n_rows = dense.size(0)
    indptr, indices = [], []
    for row in range(n_rows):
        indptr.append(len(indices))
        for col in torch.nonzero(dense[row]):
            indices.append(col.item())
    indptr.append(len(indices))
    #
    indptr = torch.IntTensor(indptr)
    indices = torch.IntTensor(indices)
    return indptr, indices


def check_gspmm():
    density = 0.1
    n_src = random.randint(1, 2048)
    n_dst = random.randint(1, 2048)
    n_features = random.randint(1, 256)

    #
    adj_adjacency = None
    while True:
        dense_raw = torch.rand(
            size=[n_dst, n_src]
        )
        adj_adjacency = torch.where(
            dense_raw < density, 1.0, 0.0
        )
        if adj_adjacency.max() != 0.0:
            break
    x = torch.randn(
        [n_src, n_features]
    )
    rev_adjacency = torch.transpose(
        adj_adjacency, 0, 1
    )
    adj_sparse = to_csr(adj_adjacency)
    rev_sparse = to_csr(rev_adjacency)
    assert torch.allclose(
        to_dense(
            n_dst, n_src,
            adj_sparse
        ),
        adj_adjacency,
        atol=1e-5, rtol=1e-5
    )

    #
    x = x.to('cuda')
    adj_sparse = [
        v.to('cuda') for v in adj_sparse
    ]
    rev_sparse = [
        v.to('cuda') for v in rev_sparse
    ]
    adj_adjacency = adj_adjacency.to('cuda')
    linear = nn.Linear(
        n_features, n_features
    ).to('cuda')

    #
    linear.zero_grad()
    y1 = torch.matmul(
        adj_adjacency,
        linear(x)
    )
    y1.sum().backward()
    grad_1 = linear.weight.grad.clone()

    #
    linear.zero_grad()
    y2 = sageir.gspmm(
        adj_sparse,
        rev_sparse,
        linear(x)
    )
    y2.sum().backward()
    grad_2 = linear.weight.grad.clone()

    #
    assert torch.allclose(
        y1, y2, atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(
        grad_1, grad_2, atol=1e-3, rtol=1e-3
    )


class GCNLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.fc = nn.Linear(
            in_features, out_features
        )

    def forward(self,
                block: sageir.Block,
                x: torch.Tensor):
        x = self.fc(x)
        x = sageir.gspmm(
            block.adj_sparse,
            block.rev_sparse,
            x
        )
        return x


def check_layer():
    dataset = CoraGraphDataset(
        verbose=False
    )

    #
    graph = dataset[0].to('cuda')
    block = sageir.from_dglgraph(graph)

    #
    n_inputs, d_hidden = 64, 16
    feature = torch.randn([
        graph.num_nodes(), n_inputs
    ]).to('cuda')
    layer1 = dglnn.GraphConv(
        n_inputs, d_hidden, 'none'
    ).to('cuda')
    layer2 = GCNLayer(
        n_inputs, d_hidden
    ).to('cuda')

    #
    with torch.no_grad():
        bias = layer1.bias
        if bias is not None:
            layer2.fc.bias.copy_(bias)
        weight = layer1.weight.T
        layer2.fc.weight.copy_(weight)

    #
    y1 = layer1(graph, feature)
    y2 = layer2(block, feature)

    #
    assert torch.allclose(
        y1, y2, atol=1e-3, rtol=1e-3
    )


def test():
    for _ in tqdm(range(256)):
        check_gspmm()
        check_layer()


if __name__ == "__main__":
    test()
