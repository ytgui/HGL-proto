import torch
import random
from torch import nn, autograd
from sageir import sparse, convert
from tqdm import tqdm


def check_gspmm():
    density = 0.02
    n_src = random.randint(1, 512)
    n_dst = random.randint(1, 512)
    n_heads = random.randint(1, 32)
    """
    density = 0.2
    n_src, n_dst, n_heads = 2, 3, 2
    """
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
    mm_sizes = [n_dst, n_features, n_src]
    indptr, indices = convert.to_csr(adj_adjacency)

    #
    x = torch.randn(
        [n_src, n_heads, n_features]
    ).to('cuda')
    values = torch.randn(
        [indices.size(0), n_heads]
    ).to('cuda')
    values.requires_grad = True
    linear = nn.Linear(
        n_features, n_features
    ).to('cuda')
    indptr = indptr.to('cuda')
    indices = indices.to('cuda')
    adj_sparse = [indptr, indices]
    attn_adjacency = convert.to_dense_mha(
        n_dst, n_src, adj_sparse, values
    ).to('cuda')
    attn_adjacency.requires_grad = True
    adj_adjacency = adj_adjacency.to('cuda')

    #
    y_1 = torch.bmm(
        attn_adjacency,
        linear(x).transpose(0, 1)
    ).transpose(0, 1)
    y_1.sum().backward()
    grad_1 = linear.weight.grad.clone()
    grad_2 = attn_adjacency.grad.clone()
    grad_2 = torch.multiply(
        adj_adjacency.unsqueeze(0), grad_2
    )

    #
    linear.zero_grad()
    y_2 = sparse.GSPMMFunction.apply(
        mm_sizes, adj_sparse, values, linear(x)
    )
    y_2.sum().backward()
    grad_3 = linear.weight.grad.clone()
    grad_4 = convert.to_dense_mha(
        n_dst, n_src, adj_sparse,
        values.grad.clone()
    ).to('cuda')

    #
    assert torch.allclose(
        y_1, y_2, atol=1e-3
    )
    assert torch.allclose(
        grad_1, grad_3, atol=1e-3
    )
    assert torch.allclose(
        grad_2, grad_4, atol=1e-3
    )


def check_gsddmm():
    density = 0.02
    n_src = random.randint(1, 512)
    n_dst = random.randint(1, 512)
    n_heads = random.randint(1, 16)
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
    mm_sizes = [n_dst, n_features, n_src]
    indptr, indices = convert.to_csr(adj_adjacency)

    #
    x_dst = torch.randn(
        [n_dst, n_heads, n_features]
    ).to('cuda')
    x_src = torch.randn(
        [n_src, n_heads, n_features]
    ).to('cuda')
    linear_q = nn.Linear(
        n_features, 1
    ).to('cuda')
    linear_k = nn.Linear(
        n_features, 1
    ).to('cuda')
    indptr = indptr.to('cuda')
    indices = indices.to('cuda')
    adj_sparse = [indptr, indices]
    adj_adjacency = adj_adjacency.to('cuda')

    #
    q = linear_q(x_dst).squeeze(-1)
    k = linear_k(x_src).squeeze(-1)
    coeff_r = k.repeat([n_dst, 1])
    coeff_l = q.repeat_interleave(n_src, dim=0)
    coeff_e = (coeff_l + coeff_r).view(
        [n_dst, n_src, -1]
    )
    coeff_e = nn.LeakyReLU(
        negative_slope=0.2
    )(coeff_e)
    negative = -1e15 * torch.ones_like(coeff_e)
    coeff_e = torch.where(
        adj_adjacency.unsqueeze(-1) > 0.0,
        coeff_e, negative
    )
    coeff_e = coeff_e.transpose(0, 2)
    coeff_e = coeff_e.transpose(1, 2)
    attn_1 = torch.softmax(coeff_e, dim=-1)
    attn_1 = torch.multiply(
        adj_adjacency.unsqueeze(0), attn_1
    )
    y_1 = torch.bmm(
        attn_1, x_src.transpose(0, 1)
    ).transpose(0, 1)
    y_1.sum().backward()
    grad_q_1 = linear_q.weight.grad.clone()
    grad_k_1 = linear_k.weight.grad.clone()

    #
    linear_q.zero_grad()
    linear_k.zero_grad()
    q = linear_q(x_dst).squeeze(-1)
    k = linear_k(x_src).squeeze(-1)
    attn_2 = sparse.GSDDMMFunction.apply(
        adj_sparse, q, k
    )
    y_2 = sparse.GSPMMFunction.apply(
        mm_sizes, adj_sparse, attn_2, x_src
    )
    y_2.sum().backward()
    grad_q_2 = linear_q.weight.grad.clone()
    grad_k_2 = linear_k.weight.grad.clone()

    #
    assert torch.allclose(
        y_1, y_2, atol=1e-3
    )
    assert torch.allclose(
        grad_q_1, grad_q_2, atol=1e-3
    )
    assert torch.allclose(
        grad_k_1, grad_k_2, atol=1e-3
    )


def test():
    for _ in tqdm(range(256)):
        check_gspmm()
        check_gsddmm()


if __name__ == "__main__":
    test()
