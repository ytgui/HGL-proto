import torch


class Block:
    def __init__(self):
        self.adj_sparse = None
        self.rev_sparse = None


def from_dglgraph(graph):
    block = Block()
    sparse = graph.adjacency_matrix(
        transpose=False, scipy_fmt='csr'
    )
    block.adj_sparse = [
        torch.IntTensor(
            sparse.indptr
        ).to(graph.device),
        torch.IntTensor(
            sparse.indices
        ).to(graph.device)
    ]
    sparse = graph.adjacency_matrix(
        transpose=True, scipy_fmt='csr'
    )
    block.rev_sparse = [
        torch.IntTensor(
            sparse.indptr
        ).to(graph.device),
        torch.IntTensor(
            sparse.indices
        ).to(graph.device)
    ]
    return block
