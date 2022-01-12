from numpy.core.fromnumeric import size
import torch
from torch import nn
from sageir import ir, trace


class Module2IR:
    def __init__(self):
        pass

    def _visit(self, node):
        if isinstance(node, (list, tuple)):
            return ir.OpTuple(prevs=[
                self._visit(x)
                for x in node
            ])
        elif isinstance(node, trace.Tracer):
            if not node.previous_func:
                return ir.OpTensor(
                    size=node.size()
                )
            elif node.previous_func == 'linear':
                node_b = None
                if 'bias' in node.previous_kwargs:
                    node_b = node.previous_kwargs['bias']
                node_x, node_w = node.previous_args
                return ir.OpLinear(
                    x=self._visit(node_x),
                    w=node_w,
                    b=node_b
                )
            elif node.previous_func == 'gspmm':
                node_a, node_r, node_x = node.previous_args
                return ir.OpGSPMM(
                    a=self._visit(node_a),
                    r=self._visit(node_r),
                    x=self._visit(node_x)
                )
            else:
                raise NotImplementedError
        elif isinstance(node, torch.Tensor):
            return ir.OpTensor(
                size=node.size()
            )
        else:
            raise NotImplementedError

    def transform(self, model: nn.Module, args: list) -> ir.Op:
        args = [
            trace.Tracer(
                torch.zeros_like(x, device='cpu')
            ).to(x.device)
            if isinstance(x, torch.Tensor) else x
            for x in args
        ]

        #
        output = model(*args)
        root = self._visit(output)
        return root
