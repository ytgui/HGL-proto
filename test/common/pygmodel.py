from torch import nn
from torch_geometric import nn as pygnn


class PyGGCNModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        self.i2h = pygnn.GCNConv(
            in_features, gnn_features
        )
        self.h2o = pygnn.GCNConv(
            gnn_features, out_features
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.i2h(x, edge_index)
        x = self.activation(x)
        x = self.h2o(x, edge_index)
        return x
