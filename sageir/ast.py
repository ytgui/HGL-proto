import torch
import sageir
from torch import nn
from sageir import mp, ir, trace


class Module2IR:
    def __init__(self):
        self._tracer2ir = dict()

    def _visit(self, node, kwargs: dict):
        #
        if node in self._tracer2ir:
            return self._tracer2ir[node]

        #
        if isinstance(node, sageir.Block):
            for k, g in kwargs.items():
                if g != node:
                    continue
                self._tracer2ir[node] = ir.OpGraph(
                    g, name=k
                )
                return self._tracer2ir[node]
            for k, het in kwargs.items():
                if not isinstance(
                    het, sageir.HeteroBlock
                ):
                    continue
                for rel, blk in \
                        het.hetero_graph.items():
                    if blk != node:
                        continue
                    self._tracer2ir[node] = ir.OpGraph(
                        blk,
                        name='{}.{}'.format(
                            k, het.rel2idx[rel]
                        )
                    )
                    return self._tracer2ir[node]
            raise RuntimeError
        elif isinstance(node, trace.Tracer):
            if not node.previous_func:
                for k, v in kwargs.items():
                    if id(v) != id(node):
                        continue
                    self._tracer2ir[node] = ir.OpTensor(
                        size=node.size(),
                        name=k
                    )
                    return self._tracer2ir[node]
                raise RuntimeError
            elif node.previous_func == 'view':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpView(
                        x=self._visit(node_x, kwargs),
                        size=node.previous_kwargs['size']
                    )
                return self._tracer2ir[node]
            elif node.previous_func == 'linear':
                node_b = None
                if 'bias' in node.previous_kwargs:
                    node_b = node.previous_kwargs['bias']
                node_x, node_w = node.previous_args
                self._tracer2ir[node] = ir.OpLinear(
                    x=self._visit(node_x, kwargs),
                    w=node_w, b=node_b
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'leaky_relu':
                node_x, = node.previous_args
                self._tracer2ir[node] = ir.OpLeakyRelu(
                    x=self._visit(node_x, kwargs)
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'reduce_wrapper':
                block, func_name = node.previous_args
                prevs = {
                    'g': self._visit(block, kwargs)
                }
                prevs.update({
                    k: self._visit(v, kwargs)
                    for k, v in node.previous_kwargs.items()
                })
                self._tracer2ir[node] = ir.OpVertFunc(
                    size=[block.num_nodes()],
                    prevs=prevs, func_name=func_name
                )
                return self._tracer2ir[node]
            elif node.previous_func == 'message_wrapper':
                block, func_name = node.previous_args
                prevs = {
                    'g': self._visit(block, kwargs)
                }
                prevs.update({
                    k: self._visit(v, kwargs)
                    for k, v in node.previous_kwargs.items()
                })
                self._tracer2ir[node] = ir.OpEdgeFunc(
                    size=[block.num_edges()],
                    prevs=prevs, func_name=func_name
                )
                return self._tracer2ir[node]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def transform(self, model: nn.Module, kwargs: dict) -> ir.Op:
        # build tracer
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
            elif isinstance(x, mp.Graph):
                return x
            else:
                raise NotImplementedError
        kwargs = process(kwargs)

        #
        output = model(**kwargs)

        # replace mp.Graph
        def post_process(kwargs):
            post_insert = []
            post_removal = []
            for k, v in kwargs.items():
                if isinstance(v, mp.Graph):
                    post_removal.append(k)
                    post_insert.append([k, v.blk])
            for k in post_removal:
                del kwargs[k]
            for k, v in post_insert:
                kwargs[k] = v
            return kwargs
        kwargs = post_process(kwargs)

        # transform to ir by tracer
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
