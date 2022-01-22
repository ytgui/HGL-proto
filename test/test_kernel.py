import time
import torch
import random
import sageir
from torch import nn, autograd
from tqdm import tqdm


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
    block = sageir.Block(
        size=[n_dst, n_src],
        adj=adj_sparse, rev=rev_sparse
    )
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
        block, linear(x)
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


def leaky_relu(x, negative_slope):
    return max(0, x) + negative_slope * min(0, x)


def grad_leaky_relu(x, negative_slope):
    if x < 0.0:
        return negative_slope
    return 1.0


class GSDDMMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, adj_sparse, q, k):
        import graph_ext
        ctx.adj_sparse = adj_sparse
        indptr, indices = adj_sparse
        attn_values = graph_ext.sddmm_forward(
            indptr, indices, q, k
        )
        #
        ctx.save_for_backward(q, k, attn_values)
        return attn_values

    @staticmethod
    def backward(ctx, grad_out):
        indptr, indices = ctx.adj_sparse
        q, k, attn_values = ctx.saved_tensors
        assert len(ctx.needs_input_grad) == 3
        assert ctx.needs_input_grad[0] is False
        assert ctx.needs_input_grad[1] is True
        assert ctx.needs_input_grad[2] is True
        #
        import graph_ext
        grad_q, grad_k = graph_ext.sddmm_backward(
            indptr, indices, q, k, attn_values, grad_out
        )
        #
        return None, grad_q, grad_k


def check_gsddmm():
    #
    n_heads = 2
    n_nodes = 3
    n_features = 2

    #
    adj_adjacency = torch.FloatTensor([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    adj_sparse = to_csr(adj_adjacency)

    #
    q = torch.randn(
        size=[n_nodes, n_heads], device='cuda',
        requires_grad=True
    )
    k = torch.randn(
        size=[n_nodes, n_heads], device='cuda',
        requires_grad=True
    )
    adj_sparse = [
        x.to('cuda') for x in adj_sparse
    ]
    adj_adjacency = adj_adjacency.to('cuda')

    #
    print('----- autograd -----')
    coeff_r = k.repeat([n_nodes, 1])
    coeff_l = q.repeat_interleave(n_nodes, dim=0)
    coeff_e = (coeff_l + coeff_r).view(
        [n_nodes, n_nodes, -1]
    )
    coeff_e = nn.LeakyReLU(
        negative_slope=0.2
    )(coeff_e)
    coeff_e = torch.multiply(
        adj_adjacency.unsqueeze(-1), coeff_e
    )
    negative = -9e15 * torch.ones_like(coeff_e)
    coeff_e = torch.where(
        adj_adjacency.unsqueeze(-1) > 0.0,
        coeff_e, negative
    )
    attn_score = torch.softmax(coeff_e, dim=1)
    torch.sum(attn_score[0, 0, :]).backward()
    for i in range(n_heads):
        print(attn_score[:, :, i])
        print(q.grad.clone()[:, i])
        print(k.grad.clone()[:, i])

    #
    print('----- custom fused -----')
    q.grad.zero_()
    k.grad.zero_()
    output = GSDDMMFunction.apply(
        adj_sparse, q, k
    )
    torch.sum(output[0, :]).backward()
    for i in range(n_heads):
        print(output[:, i])
        print(q.grad.clone()[:, i])
        print(k.grad.clone()[:, i])

    return


def test():
    for _ in tqdm(range(256)):
        # check_gspmm()
        check_gsddmm()


if __name__ == "__main__":
    test()
