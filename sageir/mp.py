import torch
import sageir
import dgl as dgl
from torch import overrides
from sageir import graph


def message_wrapper(n_edges, func, **kwargs):
    return torch.zeros(size=[n_edges])


class F:
    @classmethod
    def u_add_v(cls, u, v, e):
        desc = {
            'func': 'u_add_v',
            'u': u, 'v': v, 'out': e
        }
        return desc

    @classmethod
    def u_mul_e(cls, u, e, v):
        pass

    @classmethod
    def edge_softmax(cls, e1, e2):
        pass

    @classmethod
    def aggregate_sum(cls, e, v):
        pass


class Graph:
    def __init__(self, blk: graph.Block):
        self.blk = blk
        self.edge = dict()
        self.src_node = dict()
        self.dst_node = dict()

    def message_func(self, desc: dict):
        kwargs = {}
        for k, v in desc.items():
            if k in ['func', 'out']:
                continue
            for data in [self.edge,
                         self.src_node,
                         self.dst_node]:
                if v not in data:
                    continue
                kwargs[k] = data[v]
        n_edges = self.blk.num_edges()
        self.edge[desc['out']] = \
            overrides.handle_torch_function(
            message_wrapper,
            kwargs.values(),
            n_edges, desc['func'], **kwargs
        )

    def reduce_func(self, desc):
        a = 0

    def update_func(self, desc):
        a = 0


def from_dglgraph(graph):
    import dgl
    assert isinstance(
        graph, (
            dgl.DGLGraph,
            dgl.DGLHeteroGraph
        )
    )
    return Graph(sageir.from_dglgraph(graph))
