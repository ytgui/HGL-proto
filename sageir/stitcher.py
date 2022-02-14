import torch
from sageir import mp, ir, block
from tqdm import tqdm


class Stitcher:
    def _packing_ffd(self, weights, cap):
        bins = []
        n_bins = 0
        weights = sorted(
            weights,
            key=lambda x: x[-1],
            reverse=True
        )
        bin_remains = [0] * len(weights)
        for item in weights:
            w = item[-1]
            for i in range(n_bins):
                if w <= bin_remains[i]:
                    bin_remains[i] -= w
                    bins[i].append(item)
                    break
            else:
                bin_remains[n_bins] = cap - w
                bins.append([item])
                n_bins += 1
        assert n_bins == len(bins)
        while len(bins) > 2:
            bin_sizes = [
                sum([x[-1] for x in group])
                for group in bins
            ]
            extra = bins.pop(-1)
            if bin_sizes[0] < bin_sizes[1]:
                bins[0].extend(extra)
            else:
                bins[1].extend(extra)
        return bins

    def _build_hgraph(self, hgraph: mp.HeteroGraph, stitch_rules: dict):
        stitch_map = {}
        new_hgraph = mp.HeteroGraph()
        new_hgraph.device = hgraph.device
        new_hgraph.nty2num = hgraph.nty2num
        for idx, stitches in stitch_rules.items():
            n_edges = 0
            n_src_nodes = 0
            indptr_list = []
            indices_list = []
            # stitch
            for sty, ety, dty, num in stitches:
                n_edges += num
                old_blk = hgraph.hetero_graph[
                    sty, ety, dty
                ].blk
                n_src_nodes += old_blk.num_src_nodes()
                indptr_list.append(old_blk.adj_sparse[0])
                indices_list.append(old_blk.adj_sparse[1])
            # concat
            n_rows = hgraph.nty2num[dty]
            new_indptr, new_indices = [], []
            for row in tqdm(range(n_rows)):
                new_indptr.append(len(new_indices))
                for indptr, indices in \
                        zip(indptr_list, indices_list):
                    for i in range(indptr[row], indptr[row + 1]):
                        new_indices.append(indices[i].item())
            new_indptr.append(len(new_indices))
            assert len(new_indptr) == n_rows + 1
            assert len(new_indices) == n_edges
            new_indptr = torch.IntTensor(
                new_indptr).to(hgraph.device)
            new_indices = torch.IntTensor(
                new_indices).to(hgraph.device)
            new_blk = block.Block(
                size=[n_rows, n_src_nodes],
                adj=[new_indptr, new_indices]
            )
            # new name
            new_sty = '-->'.join(
                item[0] for item in stitches
            )
            new_ety = '-->'.join(
                item[1] for item in stitches
            )
            new_hgraph.idx2rel[
                len(new_hgraph.idx2rel)
            ] = [new_sty, new_ety, dty]
            new_hgraph.rel2idx[
                new_sty, new_ety, dty
            ] = len(new_hgraph.rel2idx)
            new_hgraph.hetero_graph[
                new_sty, new_ety, dty
            ] = mp.Graph(new_blk)
            stitch_map[idx] = (
                new_sty, new_ety, dty
            )
        return new_hgraph, stitch_map

    def _replace_spmm(self,
                      spmm_nodes: set,
                      hgraph: mp.HeteroGraph,
                      stitch_rules: dict,
                      stitch_map: dict,
                      new_hgraph: mp.HeteroGraph):
        #
        def match_rule(stgt, etgt, dtgt):
            for idx, stitches in stitch_rules.items():
                for sty, ety, dty, _ in stitches:
                    if sty == stgt and \
                            ety == etgt and \
                            dty == dtgt:
                        return idx
            raise RuntimeError

        stitch_idxes = set()
        for spmm_node in spmm_nodes:
            edge_node = spmm_node.prevs['e']
            if not isinstance(edge_node, ir.OpFusedSDDMM):
                raise NotImplementedError
            graph_node = spmm_node.prevs['g']
            splited = graph_node.name.split('.')
            stitch_idxes.add(
                match_rule(
                    *hgraph.idx2rel[int(splited[1])]
                )
            )

        #
        def match_spmms(stitches):
            node_res = []
            for stgt, etgt, dtgt, _ in stitches:
                for spmm_node in spmm_nodes:
                    graph_node = spmm_node.prevs['g']
                    splited = graph_node.name.split('.')
                    sty, ety, dty = hgraph.idx2rel[int(splited[1])]
                    if sty == stgt and \
                            ety == etgt and \
                            dty == dtgt:
                        node_res.append(spmm_node)
            assert len(stitches) == len(node_res)
            return node_res

        for idx in stitch_idxes:
            stitches = stitch_rules[idx]
            if len(stitches) < 2:
                continue
            stitch_rel = stitch_map[idx]
            stitch_spmms = match_spmms(stitches)
            new_graph = new_hgraph.hetero_graph[stitch_rel]
            # replace nodes
            stitch_queries = []
            stitch_keys, stitch_values = [], []
            for spmm_node in stitch_spmms:
                graph_node = spmm_node.prevs['g']
                sddmm_node = spmm_node.prevs['e']
                if not isinstance(sddmm_node, ir.OpFusedSDDMM):
                    raise NotImplementedError
                value_node = spmm_node.prevs['x']
                query_node = sddmm_node.prevs['q']
                key_node = sddmm_node.prevs['k']
                #
                stitch_queries.append(query_node)
                stitch_values.append(value_node)
                stitch_keys.append(key_node)
            new_key = ir.OpConcat(xs=stitch_keys, dim=0)
            new_value = ir.OpConcat(xs=stitch_values, dim=0)
            new_query = ir.OpConcat(xs=stitch_queries, dim=0)
            a1 = new_graph.num_src_nodes()
            a2 = new_graph.num_dst_nodes()
            a = 0

        return

    def _stitch_hetg(self, dataflow: ir.Op, kwargs: dict):
        # group by dty
        many2one = {}
        hgraph: mp.HeteroGraph = kwargs['hgraph']
        for sty, ety, dty in hgraph.hetero_graph:
            if dty not in many2one:
                many2one[dty] = []
            blk: block.Block = hgraph.hetero_graph[
                sty, ety, dty
            ].blk
            many2one[dty].append(
                [sty, ety, dty, blk.num_edges()]
            )

        # bin packing
        stitch_rules = {}
        for het_weights in many2one.values():
            if len(het_weights) <= 2:
                continue
            bin_cap = sum(
                [x[-1] for x in het_weights]
            )
            for candidate in self._packing_ffd(
                    het_weights, cap=bin_cap // 2):
                idx = len(stitch_rules)
                stitch_rules[idx] = candidate

        # build graph
        new_hgraph, stitch_map = self._build_hgraph(
            hgraph, stitch_rules=stitch_rules
        )
        assert len(stitch_rules) == len(stitch_map)
        assert len(new_hgraph.hetero_graph) == len(stitch_map)

        #
        def visit_scale(root_node: ir.Op,
                        scale_nodes=None):
            if scale_nodes is None:
                scale_nodes = set()
            for name in root_node.prevs:
                visit_scale(root_node.prevs[name],
                            scale_nodes)
            #
            if isinstance(root_node, ir.OpScale):
                scale_nodes.add(root_node)
            return scale_nodes
        scale_nodes = visit_scale(dataflow)

        #
        def visit_spmm(root_nodes: list,
                       spmm_nodes=None,
                       depth=0):
            if depth >= 3:
                return
            if spmm_nodes is None:
                spmm_nodes = set()
            for node in root_nodes:
                if isinstance(node, ir.OpSPMM):
                    spmm_nodes.add(node)
                    continue
                for child in node.prevs.values():
                    if isinstance(child, ir.OpAdd):
                        continue
                    visit_spmm([child],
                               spmm_nodes,
                               depth + 1)
            return spmm_nodes

        def visit_accumulate(root_node: ir.Op,
                             accum_nodes=None):
            if accum_nodes is None:
                accum_nodes = set()
            for child in root_node.prevs.values():
                if isinstance(child, ir.OpAdd):
                    accum_nodes.add(child)
                    visit_accumulate(child, accum_nodes)
            return accum_nodes

        # begin stitching
        for scale_node in scale_nodes:
            accum_nodes = visit_accumulate(scale_node)
            spmm_nodes = visit_spmm(accum_nodes)
            assert len(spmm_nodes) == len(accum_nodes) + 1
            self._replace_spmm(
                spmm_nodes=spmm_nodes, hgraph=hgraph,
                stitch_rules=stitch_rules, stitch_map=stitch_map,
                new_hgraph=new_hgraph
            )
            a = 0

        raise NotImplementedError

    def transform(self, dataflow, kwargs: dict):
        if isinstance(dataflow, ir.Op):
            dataflow = self._stitch_hetg(
                dataflow, kwargs=kwargs
            )
            return dataflow
        else:
            raise NotImplementedError
