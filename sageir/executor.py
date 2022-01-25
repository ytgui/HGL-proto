import torch
from torch.nn import functional
from sageir import mp, ir, graph, sparse


class Executor:
    def __init__(self):
        pass

    def _execute(self, root_node: ir.Op, kwargs: dict):
        child_args = {
            k: self._execute(v, kwargs)
            for k, v in root_node.prevs.items()
        }
        if isinstance(root_node, ir.OpGSPMM):
            return sparse.gspmm(
                block=child_args['g'],
                edge=child_args['e'],
                x=child_args['x']
            )
        elif isinstance(root_node, ir.OpFusedSDDMM):
            return sparse.fused_gsddmm(
                block=child_args['g'],
                query=child_args['q'],
                key=child_args['k']
            )
        elif isinstance(root_node, ir.OpLinear):
            return functional.linear(
                input=child_args['x'],
                weight=root_node.ref_params['weight'],
                bias=root_node.ref_params['bias']
            )
        elif isinstance(root_node, ir.OpView):
            return child_args['x'].view(
                size=root_node.size
            )
        else:
            if root_node.name in kwargs:
                return kwargs[root_node.name]
            raise NotImplementedError

    def run(self, dataflow: ir.Op, kwargs: dict):
        # replace mp.Graph
        def process(kwargs):
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
        kwargs = process(kwargs)

        #
        return self._execute(dataflow, kwargs=kwargs)
