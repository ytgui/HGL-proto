import torch


class Block:
    def __init__(self, size:list, adj: list, right_norm=None):
        self.size = size
        self.adj_sparse = adj
        self.right_norm = right_norm

    def num_edges(self):
        indices = self.adj_sparse[1]
        assert indices.dim() == 1
        return indices.numel()

    def num_src_nodes(self):
        return self.size[1]

    def num_dst_nodes(self):
        indptr = self.adj_sparse[0]
        assert indptr.dim() == 1
        assert self.size[0] == indptr.size(0) - 1
        return self.size[0]
