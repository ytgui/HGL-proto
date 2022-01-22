import torch


class Block:
    def __init__(self, size=None,
                 adj=None, rev=None):
        self.size = size
        self.adj_sparse = adj
        self.rev_sparse = rev
    
    def num_nodes(self):
        indptr = self.adj_sparse[0]
        assert indptr.dim() == 1
        return indptr.numel() - 1
    
    def num_edges(self):
        indices = self.adj_sparse[1]
        assert indices.dim() == 1
        return indices.numel()

    @staticmethod
    def from_dglgraph(graph):
        import dgl
        assert isinstance(
            graph, dgl.DGLGraph
        )
        assert graph.is_homogeneous

        #
        adj = graph.adj(
            transpose=True,
            scipy_fmt='csr'
        )
        rev = graph.adj(
            transpose=False,
            scipy_fmt='csr'
        )
        block = Block(
            size=[
                graph.num_nodes(),
                graph.num_nodes()
            ],
            adj=[
                torch.IntTensor(
                    adj.indptr
                ).to(graph.device),
                torch.IntTensor(
                    adj.indices
                ).to(graph.device)
            ],
            rev=[
                torch.IntTensor(
                    rev.indptr
                ).to(graph.device),
                torch.IntTensor(
                    rev.indices
                ).to(graph.device)
            ]
        )
        return block


class HeteroBlock:
    def __init__(self):
        self.etypes = []
        # (sty, ety, dty): g
        self.rel2idx = {}
        self.hetero_graph = {}

    def __iter__(self):
        return iter(self.hetero_graph.items())

    @staticmethod
    def from_dglgraph(graph):
        import dgl
        assert isinstance(
            graph, dgl.DGLHeteroGraph
        )

        #
        hblock = HeteroBlock()
        hblock.etypes = list(
            graph.etypes
        )
        for sty, ety, dty in \
                graph.canonical_etypes:
            adj = graph.adj(
                transpose=True,
                scipy_fmt='csr',
                etype=(sty, ety, dty)
            )
            rev = graph.adj(
                transpose=False,
                scipy_fmt='csr',
                etype=(sty, ety, dty)
            )
            hblock.rel2idx[
                sty, ety, dty
            ] = len(hblock.rel2idx)
            hblock.hetero_graph[
                sty, ety, dty
            ] = Block(
                size=[
                    graph.num_nodes(dty),
                    graph.num_nodes(sty)
                ],
                adj=[
                    torch.IntTensor(
                        adj.indptr
                    ).to(graph.device),
                    torch.IntTensor(
                        adj.indices
                    ).to(graph.device)
                ],
                rev=[
                    torch.IntTensor(
                        rev.indptr
                    ).to(graph.device),
                    torch.IntTensor(
                        rev.indices
                    ).to(graph.device)
                ]
            )
        return hblock


def from_dglgraph(graph):
    import dgl
    assert isinstance(
        graph, (
            dgl.DGLGraph,
            dgl.DGLHeteroGraph
        )
    )
    if graph.is_homogeneous:
        return Block.from_dglgraph(graph)
    else:
        return HeteroBlock.from_dglgraph(graph)
