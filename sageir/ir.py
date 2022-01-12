import torch
from torch import nn
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


class OpTensor(Op):
    def __init__(self, size: List[int], prevs: dict = {}, name: str = ''):
        Op.__init__(self, prevs, name)
        assert isinstance(size, (list, tuple))
        self.size = size
        if not prevs:
            return
        assert isinstance(prevs, dict)
        for n in prevs.values():
            n.next = self


class OpTuple(Op):
    def __init__(self, prevs: List[Op], name: str = ''):
        Op.__init__(self, prevs, name)
        assert len(prevs) > 0
        prevs = {
            i: x
            for i, x in enumerate(prevs)
        }
        self.size = [len(prevs)]
        assert isinstance(prevs, dict)
        for n in prevs.values():
            n.next = self


class OpLinear(OpTensor):
    def __init__(self,
                 x: OpTensor,
                 w: nn.Parameter,
                 b: nn.Parameter,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=[-1, 1],
            prevs={'x': x},
            name=name
        )
        self.ref_params = {'weight': w, 'bias': b}


class OpGSPMM(OpTensor):
    def __init__(self,
                 a: OpTensor,
                 r: OpTensor,
                 x: OpTensor,
                 name: str = ''):
        OpTensor.__init__(
            self,
            size=[-1, 1],
            prevs={'a': a, 'r': r, 'x': x},
            name=name
        )
