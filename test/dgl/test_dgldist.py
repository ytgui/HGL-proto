import time
import torch
from torch import nn
import dgl
import dgl.nn as dgl_nn
import dgl.data as dgl_data
import dgl.dataloading as dgl_loader
import dgl.distributed


def check_sage_training(dataset):
    #
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('device: {}'.format(device))

    #
    graph = dataset[0]
    graph.create_formats_()
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    train_mask = graph.ndata['train_mask']
    val_nodes = torch.nonzero(val_mask).squeeze()
    test_nodes = torch.nonzero(test_mask).squeeze()
    train_nodes = torch.nonzero(train_mask).squeeze()
    print('dataset split: {}-{}-{}'.format(
        len(train_nodes), len(val_nodes), len(test_nodes)))

    #
    batch_size = 32
    sampler_neighboors = 16
    sampler = dgl_loader.MultiLayerNeighborSampler(
        [sampler_neighboors] * 2
    )
    dataloader = dgl_loader.NodeDataLoader(
        graph, train_nodes, sampler,
        device=device, batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=0
    )


def test():
    # dataset
    dataset = dgl_data.RedditDataset(
        verbose=False
    )

    # profile
    import pstats
    import cProfile
    profiler = cProfile.Profile()

    #
    profiler.enable()
    check_sage_training(dataset)
    profiler.disable()
    pstats.Stats(profiler).strip_dirs() \
        .sort_stats(pstats.SortKey.TIME).print_stats(20)


if __name__ == "__main__":
    test()
