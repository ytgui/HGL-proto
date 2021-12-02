import dgl
import dglke
import argparse
from dglke import dataloader, models

#
# from dglke import train
# train.main()

#
parser = argparse.ArgumentParser()
parser.add_argument(
    '--lr', type=float, default=1e-2
)
parser.add_argument(
    '--gpu', type=int, default=[-1], nargs='+',
)
parser.add_argument(
    '--dataset', type=str, default='/home/bosskwei/.dglke/'
)
parser.add_argument(
    '--mix_cpu_gpu', type=bool, default=False
)
parser.add_argument(
    '--neg_deg_sample', type=bool, default=False
)
parser.add_argument(
    '--soft_rel_part', type=bool, default=False
)
parser.add_argument(
    '--strict_rel_part', type=bool, default=False
)
parser.add_argument(
    '--has_edge_importance', type=bool, default=False
)
parser.add_argument(
    '--regularization_coef', type=float, default=2e-6
)
parser.add_argument(
    '--regularization_norm', type=float, default=3
)
parser.add_argument(
    '--neg_adversarial_sampling', type=bool, default=False
)
args = parser.parse_args()

#
dataset = dataloader.KGDatasetFB15k(
    path=args.dataset
)
train_loader = dataloader.TrainDataset(
    dataset=dataset, args=args, ranks=1
)
head_sampler = train_loader.create_sampler(
    mode='head',
    batch_size=1024,
    neg_sample_size=256,
    neg_chunk_size=256,
    num_workers=1
)
tail_sampler = train_loader.create_sampler(
    mode='tail',
    batch_size=1024,
    neg_sample_size=256,
    neg_chunk_size=256,
    num_workers=1
)
train_sampler = dataloader.NewBidirectionalOneShotIterator(
    dataloader_head=head_sampler,
    dataloader_tail=tail_sampler,
    neg_sample_size=256, neg_chunk_size=256,
    is_chunked=True, num_nodes=dataset.n_entities,
    has_edge_importance=args.has_edge_importance
)

#
model = models.KEModel(
    args=args,
    model_name='TransE',
    n_entities=dataset.n_entities,
    n_relations=dataset.n_relations,
    hidden_dim=400, gamma=12.0
)

for i in range(200):
    pos_g, neg_g = next(train_sampler)
    loss, log = model.forward(pos_g, neg_g)
    loss.backward()
    model.update()
    print('loss:', loss.item())
a = 0
