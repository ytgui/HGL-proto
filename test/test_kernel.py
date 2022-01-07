import torch
import random
import graph_ext
from torch import autograd, nn
from tqdm import tqdm


class SPMVFunction(autograd.Function):
    @staticmethod
    def forward(ctx, adj_sparse, rev_sparse, x):
        adj_values = adj_sparse[0]
        adj_indptr = adj_sparse[1]
        adj_indices = adj_sparse[2]
        #
        y = graph_ext.spmm(
            adj_values,
            adj_indptr,
            adj_indices,
            x
        )
        #
        ctx.rev_sparse = rev_sparse
        return y

    @staticmethod
    def backward(ctx, grad_out):
        rev_values = ctx.rev_sparse[0]
        rev_indptr = ctx.rev_sparse[1]
        rev_indices = ctx.rev_sparse[2]
        grad_out = grad_out.contiguous()
        #
        grad_x = graph_ext.spmm(
            rev_values,
            rev_indptr,
            rev_indices,
            grad_out,
        )
        #
        return None, None, grad_x


sparse_spmv = SPMVFunction.apply


def to_dense(n_rows, n_cols, sparse):
    values, indptr, indices = sparse
    dense = torch.zeros(size=[n_rows, n_cols])
    for row in range(n_rows):
        for i in range(indptr[row], indptr[row + 1]):
            col = indices[i]
            val = values[i]
            dense[row, col] = val
    return dense


def to_csr(dense):
    n_rows, n_cols = dense.size()
    values, indptr, indices = [], [], []
    for row in range(n_rows):
        indptr.append(len(indices))
        for col in torch.nonzero(dense[row]):
            assert torch.is_nonzero(dense[row, col])
            values.append(dense[row, col].item())
            indices.append(col.item())
    indptr.append(len(indices))
    #
    indptr = torch.IntTensor(indptr)
    indices = torch.IntTensor(indices)
    values = torch.FloatTensor(values)
    return values, indptr, indices


def check_spmm():
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
        dense_zero = torch.zeros_like(
            dense_raw
        )
        adj_adjacency = torch.where(
            dense_raw < density,
            dense_raw, dense_zero
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
    y2 = sparse_spmv(
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


def test():
    for _ in tqdm(range(256)):
        check_spmm()


if __name__ == "__main__":
    test()
