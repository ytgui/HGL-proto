from sageir import ir


class Printer:
    def dump(self, dataflow):
        count = [-1]
        printed = set()

        #
        def visit_dfs(node: ir.Op):
            #
            if node.name and \
                    node.name in printed:
                return node.name

            #
            params = []
            for k in node.prevs:
                nid = visit_dfs(
                    node.prevs[k]
                )
                params.append(
                    '{}: %{}'.format(k, nid)
                )
            if isinstance(node, (ir.OpVertFunc,
                                 ir.OpEdgeFunc)):
                params.insert(
                    1, 'fn: {}'.format(node.func_name)
                )

            #
            if node.name:
                printed.add(node.name)
                nstr = '%' + node.name
            else:
                count[0] += 1
                nstr = '%{}'.format(count[0])
            if node.size:
                nstr += ': {}'.format(
                    node.size
                )
            nstr += ' = {}'.format(
                type(node).__name__
            )
            if len(params) > 0:
                nstr += '('
                nstr += ', '.join(params)
                nstr += ')'
            print(nstr)

            if node.name:
                return node.name
            return count[0]

        #
        if isinstance(dataflow, dict):
            for k in dataflow:
                visit_dfs(dataflow[k])
        elif isinstance(dataflow, ir.Op):
            visit_dfs(dataflow)
        else:
            raise NotImplementedError
