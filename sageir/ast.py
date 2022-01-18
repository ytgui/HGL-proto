import torch
import sageir
from torch import nn
from sageir import mp, ir, trace


class Module2IR:
    def __init__(self):
        pass

    def _visit(self, node, kwargs):
        if isinstance(node, (int, float)):
            return ir.OpTensor(
                size=[1]
            )
        elif isinstance(node, sageir.Block):
            for k, g in kwargs.items():
                if g != node:
                    continue
                return ir.OpGraph(g, name=k)
            for k, het in kwargs.items():
                if not isinstance(
                    het, sageir.HeteroBlock
                ):
                    continue
                for rel, blk in \
                        het.hetero_graph.items():
                    if blk != node:
                        continue
                    return ir.OpGraph(
                        blk,
                        name='{}.{}'.format(
                            k, het.rel2idx[rel]
                        )
                    )
            raise RuntimeError
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
                    x=self._visit(node_x, kwargs),
                    w=node_w, b=node_b
                )
            elif node.previous_func == 'gspmm':
                node_b, node_x = node.previous_args
                return ir.OpGSPMM(
                    b=self._visit(node_b, kwargs),
                    x=self._visit(node_x, kwargs)
                )
            elif node.previous_func == 'div':
                node_a, node_b = node.previous_args
                if isinstance(node_a, (int, float)):
                    node_a = 1.0 / node_a
                    return ir.OpMul(
                        a=self._visit(node_a, kwargs),
                        b=self._visit(node_b, kwargs)
                    )
                if isinstance(node_b, (int, float)):
                    node_b = 1.0 / node_b
                    return ir.OpMul(
                        a=self._visit(node_b, kwargs),
                        b=self._visit(node_a, kwargs)
                    )
                else:
                    raise NotImplementedError
            elif node.previous_func == 'add':
                node_a, node_b = node.previous_args
                return ir.OpAdd(
                    a=self._visit(node_a, kwargs),
                    b=self._visit(node_b, kwargs)
                )
            elif node.previous_func == 'zeros':
                node_a, = node.previous_args
                return ir.OpTensor(
                    size=node_a.size()
                )
            elif node.previous_func == 'leaky_relu':
                node_x, = node.previous_args
                return ir.OpLeakyRelu(
                    x=self._visit(node_x, kwargs)
                )
            elif node.previous_func == 'message_wrapper':
                n_edges, func_name = node.previous_args
                return ir.OpMessageFunc(
                    size=[n_edges],
                    prevs={
                        k: self._visit(v, kwargs)
                        for k, v in node.previous_kwargs.items()
                    },
                    func_name=func_name
                )
            else:
                raise NotImplementedError
        else:
            if isinstance(node, torch.Tensor):
                return ir.OpTensor(
                    size=node.size()
                )
            raise NotImplementedError

    def transform(self, model: nn.Module, kwargs: dict) -> ir.Op:
        def process(x):
            if isinstance(x, dict):
                return {
                    k: process(v)
                    for k, v in x.items()
                }
            elif isinstance(x, (list, tuple)):
                return [
                    process(v) for v in x
                ]
            elif isinstance(x, torch.Tensor):
                return trace.Tracer(
                    torch.zeros_like(x, device='cpu')
                ).to(x.device)
            elif isinstance(x, (mp.Graph,
                                sageir.Block,
                                sageir.HeteroBlock)):
                return x
            else:
                raise NotImplementedError
        kwargs = process(kwargs)

        #
        output = model(**kwargs)
        if isinstance(output, dict):
            root = {
                k: self._visit(
                    v, kwargs
                )
                for k, v in output.items()
            }
        elif isinstance(output, torch.Tensor):
            root = self._visit(
                output, kwargs
            )
        else:
            raise NotImplementedError

        return root
