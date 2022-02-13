import torch
from sageir import utils
from dgl.data.rdf import AIFBDataset, MUTAGDataset


def test():
    dataset = MUTAGDataset(
        verbose=True
    )
    dglgraph = dataset[0]

    #
    for sty, ety, dty in dglgraph.canonical_etypes:
        n_src = dglgraph.number_of_nodes(sty)
        n_dst = dglgraph.number_of_nodes(dty)
        n_edge = dglgraph.num_edges((sty, ety, dty))
        print('{}-->{}-->{}: [{}, {}], [{}]'.format(
            sty, ety, dty, n_dst, n_src, n_edge
        ))

    #
    a = 0


if __name__ == "__main__":
    test()
