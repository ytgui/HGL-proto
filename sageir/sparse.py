import graph_ext
from torch import autograd


class GSPMMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, adj_sparse, rev_sparse, x):
        adj_indptr = adj_sparse[0]
        adj_indices = adj_sparse[1]
        #
        y = graph_ext.gspmm(
            adj_indptr,
            adj_indices,
            x
        )
        #
        ctx.rev_sparse = rev_sparse
        return y

    @staticmethod
    def backward(ctx, grad_out):
        rev_indptr = ctx.rev_sparse[0]
        rev_indices = ctx.rev_sparse[1]
        grad_out = grad_out.contiguous()
        #
        grad_x = graph_ext.gspmm(
            rev_indptr,
            rev_indices,
            grad_out,
        )
        #
        return None, None, grad_x


gspmm = GSPMMFunction.apply
