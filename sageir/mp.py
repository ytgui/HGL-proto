import torch
import sageir
import dgl as dgl
from torch import overrides
from sageir import graph


def message_wrapper(block: graph.Block, func, **kwargs):
    return torch.zeros(size=[block.num_edges()])


def reduce_wrapper(block: graph.Block, func, **kwargs):
    return torch.zeros(size=[block.num_nodes()])


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
        desc = {
            'func': 'u_mul_e',
            'u': u, 'e': e, 'out': v
        }
        return desc

    @classmethod
    def edge_softmax(cls, e1, e2):
        desc = {
            'func': 'edge_softmax',
            'e': e1, 'out': e2
        }
        return desc

    @classmethod
    def aggregate_sum(cls, e, v):
        desc = {
            'func': 'aggregate_sum',
            'e': e, 'out': v
        }
        return desc


class Graph:
    def __init__(self, blk: graph.Block):
        self.blk = blk
        self.edge = dict()
        self.src_node = dict()
        self.dst_node = dict()

    def _build_kwargs(self, desc: dict):
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
        return kwargs

    def message_func(self, desc: dict):
        kwargs = self._build_kwargs(desc)
        self.edge[desc['out']] = \
            overrides.handle_torch_function(
            message_wrapper, kwargs.values(),
            self.blk, desc['func'], **kwargs
        )

    def reduce_func(self, desc):
        kwargs = self._build_kwargs(desc)
        self.dst_node[desc['out']] = \
            overrides.handle_torch_function(
            reduce_wrapper, kwargs.values(),
            self.blk, desc['func'], **kwargs
        )


def from_dglgraph(graph):
    import dgl
    assert isinstance(
        graph, (
            dgl.DGLGraph,
            dgl.DGLHeteroGraph
        )
    )
    return Graph(sageir.from_dglgraph(graph))
