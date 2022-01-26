import torch
import sageir
from torch import overrides
from sageir import graph


def message_wrapper(graph, func, **kwargs):
    n_heads = None
    for v in kwargs.values():
        assert v.dim() in [2, 3]
        if not n_heads:
            n_heads = v.size(1)
        assert n_heads == v.size(1)
    return torch.zeros(
        size=[graph.num_edges(), n_heads],
        device=graph.device()
    )


def reduce_wrapper(graph, func, **kwargs):
    n_heads = None
    for v in kwargs.values():
        assert v.dim() in [2]
        if not n_heads:
            n_heads = v.size(1)
        assert n_heads == v.size(1)
    return torch.zeros(
        size=[graph.num_dst_nodes(), n_heads, graph.num_features()],
        device=graph.device())


class Fn:
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

    def device(self):
        for v in self.src_node.values():
            return v.device
        raise RuntimeError

    def num_edges(self):
        return self.blk.num_edges()

    def num_src_nodes(self):
        return self.blk.num_src_nodes()

    def num_dst_nodes(self):
        return self.blk.num_dst_nodes()

    def num_features(self):
        # TODO: fix workaround
        n_features = None
        for v in self.src_node.values():
            if v.dim() != 3:
                continue
            if not n_features:
                n_features = v.size(-1)
            assert n_features == v.size(-1)
        assert n_features
        return n_features

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
            self, desc['func'], **kwargs
        )

    def reduce_func(self, desc):
        kwargs = self._build_kwargs(desc)
        self.dst_node[desc['out']] = \
            overrides.handle_torch_function(
            reduce_wrapper, kwargs.values(),
            self, desc['func'], **kwargs
        )


def from_dglgraph(graph):
    return Graph(sageir.from_dglgraph(graph))
