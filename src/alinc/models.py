import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv
)

def load_model(key, *args, **kwargs):
    model_dict = {
        "gat": GAT
    }
    return model_dict[key.lower()](*args, **kwargs)


class MLPReadout(nn.Module):

    def __init__(self, in_dim, out_dim, n_layers=2): #L=nb_hidden_layers
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        list_fc_layers = [
            nn.Linear(in_dim//2**l, in_dim//2**(l+1), bias=True) 
            for l in range(n_layers) 
        ]
        list_fc_layers.append(
            nn.Linear(in_dim//2**n_layers, out_dim, bias=True)
        )
        self.fc_layers = nn.ModuleList(list_fc_layers)
        
        
    def forward(self, x):
        y = x
        for l in range(self.n_layers):
            y = self.fc_layers[l](y)
            y = F.relu(y)
        y = self.fc_layers[self.n_layers](y)
        return y

    
class BaseModel(nn.Module):
    def __init__(
            self,  in_dim, hidden_dim, n_classes, out_dim=None, n_layers=2, 
            edge_dim=None, in_feat_dropout=0.0, dropout=0.0, batch_norm=True, 
            residual=True, activation=F.relu, seed=42, 
            device=torch.device("cpu"), **kwargs
        ):
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if out_dim is None:
            self.out_dim = hidden_dim
        else:
            self.out_dim = out_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.edge_dim = edge_dim

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.activation = activation
        self.device = device

        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.embedding_x = nn.Embedding(self.in_dim, self.hidden_dim)
        self.mlp_layer = MLPReadout(self.out_dim, self.n_classes)
        

    def forward(self, x, edge_index, edge_attr=None):

        x = torch.argmax(x, dim=1)
        x = self.embedding_x(x)
        if self.training:
            x = self.in_feat_dropout(x)

        for i, conv in enumerate(self.convs):
            x_in = x # for residual connection

            if self.edge_dim:
                x = conv(
                    x, edge_index, edge_attr=edge_attr
                )
            else:
                x = conv(
                    x, edge_index
                )

            if self.batch_norm:
                x = self.batchnorms[i](x)

            if self.activation:
                x = self.activation(x)

            if self.residual:
                x = x_in + x # residual connection
        
        x = self.mlp_layer(x)

        return x
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss


class GAT(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, n_heads=1, 
            self_loops=False, **kwargs
        ):
        super(GAT, self).__init__(
            in_dim, hidden_dim, n_classes, out_dim=hidden_dim * n_heads, 
            **kwargs
        )
        self.n_heads = n_heads
        self.self_loops = self_loops    
        self.embedding_x = nn.Embedding(
            self.in_dim, self.hidden_dim * self.n_heads
        )
        in_channels = self.hidden_dim * self.n_heads
        for _ in range(self.n_layers):
            self.convs.append(
                GATConv(
                    in_channels, self.hidden_dim, heads=self.n_heads, 
                    edge_dim=self.edge_dim, concat=True, dropout=self.dropout,
                    add_self_loops=self.self_loops
                )
            )
            self.batchnorms.append(
                nn.BatchNorm1d(self.hidden_dim * self.n_heads)
            )
            in_channels = self.hidden_dim * self.n_heads