import torch
from torch import nn
from sageir import graph
from typing import List, Union


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
    def __init__(self, g: graph.Block, name: str):
        Op.__init__(self, {}, name)
        assert g.size and len(g.size)
        self.size = g.size


class OpTensor(Op):
    def __init__(self, size: List[int], prevs: dict = {}, name: str = ''):
        Op.__init__(self, prevs, name)
        self.size = list(size)
        if not prevs:
            return
        assert isinstance(prevs, dict)
        for n in prevs.values():
            n.next = self


"""
class OpTuple(Op):
    def __init__(self, prevs: List[Op], name: str = ''):
        assert len(prevs) > 0
        prevs = {
            i: x
            for i, x in enumerate(prevs)
        }
        Op.__init__(self, prevs, name)
        self.size = [len(prevs)]
        for n in prevs.values():
            n.next = self
"""

class OpAdd(OpTensor):
    def __init__(self,
                 a: OpTensor,
                 b: OpTensor,
                 name: str = ''):
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


class OpGSPMM(OpTensor):
    def __init__(self,
                 b: OpGraph,
                 x: OpTensor,
                 name: str = ''):
        assert len(x.size) == 2
        OpTensor.__init__(
            self,
            size=[b.size[0], x.size[1]],
            prevs={'b': b, 'x': x},
            name=name
        )
