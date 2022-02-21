import torch
import random
from torch import nn
from sageir import sparse, bundle, convert
from tqdm import tqdm


def check_gspmm():
    density = 0.02
    n_src = random.randint(1, 512)
    n_dst = random.randint(1, 512)
    n_heads = random.randint(1, 32)
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
    indptr, indices = convert.to_csr(
        adj_adjacency
    )[0]

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
        adj_sparse, values, linear(x)
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
    indptr, indices = convert.to_csr(
        adj_adjacency
    )[0]

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
        adj_sparse, attn_2, x_src
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


def check_gemm():
    n_nodes = random.randint(8, 1024)
    n_features = random.randint(8, 256)
    x = torch.randn(
        [n_nodes, n_features]
    ).to('cuda')
    x.requires_grad = True

    #
    d_hidden = random.randint(1, 32)
    fc_1 = nn.Linear(
        n_features, d_hidden, bias=False
    ).to('cuda')
    fc_2 = nn.Linear(
        n_features, d_hidden, bias=False
    ).to('cuda')
    torch.cuda.synchronize()

    #
    import time
    before = time.time()
    y_1, y_2 = fc_1(x), fc_2(x)
    torch.sum(y_1 + y_2).backward()
    torch.cuda.synchronize()
    t_1 = time.time() - before
    grad_x_1 = x.grad.clone()
    grad_fc_1 = fc_1.weight.grad.clone()
    grad_fc_2 = fc_2.weight.grad.clone()

    #
    x.grad.zero_()
    fc_1.zero_grad()
    fc_2.zero_grad()
    before = time.time()
    y_3, y_4 = bundle.GEMMBundleFunction.apply(
        x, fc_1.weight, fc_2.weight,
        fc_1.bias, fc_2.bias
    )
    torch.sum(y_3 + y_4).backward()
    torch.cuda.synchronize()
    t_2 = time.time() - before
    grad_x_2 = x.grad.clone()
    grad_fc_3 = fc_1.weight.grad.clone()
    grad_fc_4 = fc_2.weight.grad.clone()
    
    #
    assert torch.allclose(y_1, y_3, atol=1e-3)
    assert torch.allclose(y_2, y_4, atol=1e-3)
    assert torch.allclose(grad_fc_1, grad_fc_3, atol=1e-3)
    assert torch.allclose(grad_fc_2, grad_fc_4, atol=1e-3)
    assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)
    # print('{:.2f}'.format(t_2 / t_1))

    return


def test():
    a = torch.zeros(size=[1, 1])
    v1 = a.stride(0)
    v2 = a.stride(1)
    for _ in tqdm(range(65536)):
        # check_gspmm()
        # check_gsddmm()
        check_gemm()


if __name__ == "__main__":
    test()
