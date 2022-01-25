import torch
import graph_ext
from sageir import graph
from torch import autograd, overrides


class GSPMMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, adj_sparse, adj_values, x):
        indptr = adj_sparse[0]
        indices = adj_sparse[1]
        #
        y = graph_ext.spmm_forward(
            adj_values, indptr, indices,
            x
        )
        ctx.adj_sparse = adj_sparse
        ctx.save_for_backward(adj_values, x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        indptr = ctx.adj_sparse[0]
        indices = ctx.adj_sparse[1]
        values, x = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        assert len(ctx.needs_input_grad) == 3
        assert ctx.needs_input_grad[0] is False
        #
        grad_a, grad_x = graph_ext.spmm_backward(
            values, indptr, indices,
            x, grad_out
        )
        #
        return None, grad_a, grad_x


def gspmm(block: graph.Block,
          edge: torch.Tensor,
          x: torch.Tensor):
    return GSPMMFunction.apply(
        block.adj_sparse, edge, x
    )


class GSDDMMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, adj_sparse, query, key):
        indptr, indices = adj_sparse
        attn_values = graph_ext.sddmm_forward(
            indptr, indices, query, key
        )
        #
        ctx.adj_sparse = adj_sparse
        ctx.save_for_backward(query, key, attn_values)
        return attn_values

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        indptr, indices = ctx.adj_sparse
        query, key, attn_values = ctx.saved_tensors
        assert len(ctx.needs_input_grad) == 3
        assert ctx.needs_input_grad[0] is False
        assert ctx.needs_input_grad[1] is True
        assert ctx.needs_input_grad[2] is True
        #
        grad_query, grad_key = graph_ext.sddmm_backward(
            indptr, indices, query, key,
            attn_values, grad_out
        )
        #
        return None, grad_query, grad_key


def fused_gsddmm(block: graph.Block,
                 query: torch.Tensor,
                 key: torch.Tensor):
    return GSDDMMFunction.apply(
        block.adj_sparse, query, key
    )
