import torch
from torch import nn
from sageir import block
from typing import List, Dict, Union


class Op:
    def __init__(self, prevs: dict, name: str = ''):
        if type(self) is Op:
            raise RuntimeError('base class should be inherited')
        self.name = name
        self.size = None
        self.prevs = prevs
        self.next = None
        self.ref_params = {}
        self.val_params = {}


class OpGraph(Op):
    def __init__(self, blk: block.Block, name: str):
        Op.__init__(self, {}, name)
        assert blk.size
        self.size = blk.size


class OpTensor(Op):
    def __init__(self, size: List[int], prevs: dict = {}, name: str = ''):
        Op.__init__(self, prevs, name)
        self.size = list(size)
        if not prevs:
            return
        assert isinstance(prevs, dict)
        for n in prevs.values():
            n.next = self


class OpAdd(OpTensor):
    def __init__(self,
                 a: OpTensor,
                 b: OpTensor,
                 name: str = ''):
        assert a.size == b.size
        OpTensor.__init__(
            self,
            size=b.size,
            prevs={'a': a, 'b': b},
            name=name
        )


class OpMul(OpTensor):
    def __init__(self,
                 a: float,
                 b: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=b.size,
            prevs={'a': a, 'b': b},
            name=name
        )


class OpView(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 size: list,
                 name: str = ''):
        size = list(size)
        OpTensor.__init__(
            self,
            size=size,
            prevs={'x': x},
            name=name
        )


class OpMean(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 dim: int,
                 name: str = ''):
        assert dim < len(x.size)
        size = x.size[:dim] + x.size[dim+1:]
        OpTensor.__init__(
            self,
            size=size,
            prevs={'x': x},
            name=name
        )
        self.val_params['dim'] = dim


class OpScale(OpTensor):
    def __init__(self,
                 scale: float,
                 x: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )
        self.val_params['scale'] = scale


class OpLinear(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 w: nn.Parameter,
                 b: nn.Parameter,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=[x.size[0], w.size(0)],
            prevs={'x': x},
            name=name
        )
        self.ref_params = {'weight': w, 'bias': b}


class OpDropout(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 p: float,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )
        self.val_params = {'p': p}


class OpGSPMM(OpTensor):
    def __init__(self,
                 graph: OpGraph,
                 edge: OpTensor,
                 x: OpTensor,
                 name: str = ''):
        assert len(edge.size) == 2
        assert len(x.size) == 3
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'g': graph, 'e': edge, 'x': x},
            name=name
        )


class OpFusedSDDMM(OpGraph):
    def __init__(self,
                 size: list,
                 graph: OpGraph,
                 query: OpTensor,
                 key: OpTensor,
                 fusion_scheme: str,
                 name: str = ''):
        assert len(graph.size) == 2
        assert len(query.size) == 2
        assert len(key.size) == 2
        assert query.size[1] == key.size[1]
        assert graph.size[0] == query.size[0]
        assert graph.size[1] == key.size[0]
        OpTensor.__init__(
            self,
            size=size,
            prevs={'g': graph, 'q': query, 'k': key},
            name=name
        )
        self.fusion_scheme = fusion_scheme


class OpELU(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )


class OpLeakyRelu(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=x.size,
            prevs={'x': x},
            name=name
        )


class OpVertFunc(OpTensor):
    def __init__(self, size: list, prevs: dict, func_name: str):
        OpTensor.__init__(
            self,
            size=size,
            prevs=prevs
        )
        self.func_name = func_name


class OpEdgeFunc(OpTensor):
    def __init__(self, size: list, prevs: dict, func_name: str):
        OpTensor.__init__(
            self,
            size=size,
            prevs=prevs
        )
        self.func_name = func_name
