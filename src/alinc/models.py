import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATConv,
    ResGatedGraphConv
)
from torch_geometric.nn.models import MLP

def load_model(key, *args, **kwargs):
    model_dict = {
        "gcn": GCN,
        "gin": GIN,
        "gat": GAT,
        "gatedgcn": GatedGCN,
        "gps": GPS
    }
    return model_dict[key.lower()](*args, **kwargs)


class MLPReadout(nn.Module):
    """See github.com/graphdeeplearning/benchmarking-gnns
    """
    def __init__(self, in_dim, out_dim, n_layers=2):
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
            residual=True, activation=F.relu, self_loops=False, seed=42, 
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
        self.self_loops = self_loops    
        self.device = device

        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.embedding_x = nn.Embedding(self.in_dim, self.hidden_dim)
        self.readout_layer = MLPReadout(self.out_dim, self.n_classes)

    def forward(self, x, edge_index, edge_attr=None):

        x = torch.argmax(x, dim=1)
        x = self.embedding_x(x)
        if self.training:
            x = self.in_feat_dropout(x)

        x = self._conv(x, edge_index, edge_attr=edge_attr)
        
        x = self.readout_layer(x)

        return x
    
    def reset_state(self):
        self.cpu()
        self.load_state_dict(self.init_model_state)
    
    def _conv(self, x, edge_index, edge_attr=None):

        for i, conv in enumerate(self.convs):
            x_in = x # for residual connection

            if self.edge_dim:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if self.batch_norm:
                x = self.batchnorms[i](x)

            if self.activation:
                x = self.activation(x)

            if self.residual:
                x = x_in + x # residual connection

        return x
    
    def loss(self, pred, label):
        """See github.com/graphdeeplearning/benchmarking-gnns
        """
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


class GCN(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, **kwargs
        ):
        super(GCN, self).__init__(in_dim, hidden_dim, n_classes, **kwargs)

        self.dropout = nn.Dropout(self.dropout)

        in_channels = self.hidden_dim
        for _ in range(self.n_layers - 1):
            self.convs.append(
                GCNConv(
                    in_channels, self.hidden_dim, add_self_loops=self.self_loops
                )
            )
            if self.batch_norm:
                self.batchnorms.append(
                    nn.BatchNorm1d(self.hidden_dim)
                )
            in_channels = self.hidden_dim

        self.convs.append(
            GCNConv(
                in_channels, self.out_dim, add_self_loops=self.self_loops
            )
        )
        if self.batch_norm:
            self.batchnorms.append(
                nn.BatchNorm1d(self.out_dim)
            )
        self.init_model_state = copy.deepcopy(self.state_dict())

    def _conv(self, x, edge_index, edge_attr=None):

        for i, conv in enumerate(self.convs):
            x_in = x # for residual connection

            x = conv(x, edge_index)

            if self.batch_norm:
                x = self.batchnorms[i](x)

            if self.activation:
                x = self.activation(x)

            if self.residual:
                x = x_in + x # residual connection
            
            if self.training:
                x = self.dropout(x)

        return x
    

class GIN(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, train_eps=True, 
            n_layers_mlp_gin=2, **kwargs
        ):
        super(GIN, self).__init__(in_dim, hidden_dim, n_classes, **kwargs)

        self.dropout = nn.Dropout(self.dropout)
        self.train_eps = train_eps
        self.n_layers_mlp_gin = n_layers_mlp_gin

        in_channels = self.hidden_dim
        for _ in range(self.n_layers - 1):
            mlp = MLP(
                in_channels=in_channels, 
                hidden_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_layers=self.n_layers_mlp_gin
            )
            self.convs.append(
                GINConv(nn=mlp, train_eps=self.train_eps)
            )
            if self.batch_norm:
                self.batchnorms.append(
                    nn.BatchNorm1d(self.hidden_dim)
                )
            in_channels = self.hidden_dim

        
        mlp = MLP(
            in_channels=in_channels, 
            hidden_channels=self.hidden_dim,
            out_channels=self.out_dim,
            num_layers=self.n_layers_mlp_gin
        )
        self.convs.append(
                GINConv(nn=mlp, train_eps=self.train_eps)
        )
        if self.batch_norm:
            self.batchnorms.append(
                nn.BatchNorm1d(self.out_dim)
            )

        # Readout per layer
        self.readout_layer = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.n_classes)
            for _ in range(self.n_layers)
        ])
        self.readout_layer.append(
            nn.Linear(self.out_dim, self.n_classes)
        )
        self.init_model_state = copy.deepcopy(self.state_dict())

    def forward(self, x, edge_index, edge_attr=None):

        x = torch.argmax(x, dim=1)
        x = self.embedding_x(x)

        if self.training:
            x = self.in_feat_dropout(x)

        x = self._conv(x, edge_index)
        
        score_over_layer = 0
        for i, xi in enumerate(x):
            score_over_layer += self.readout_layer[i](xi)

        return score_over_layer

    def _conv(self, x, edge_index):

        x_layers = [x]

        for i, conv in enumerate(self.convs):
            x_in = x # for residual connection

            x = conv(x, edge_index)

            if self.batch_norm:
                x = self.batchnorms[i](x)

            if self.activation:
                x = self.activation(x)

            if self.residual:
                x = x_in + x # residual connection
            
            if self.training:
                x = self.dropout(x)

            x_layers.append(x)

        return x_layers


class GAT(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, n_heads=1, **kwargs
        ):
        out_dim = kwargs.pop("out_dim", None)
        if out_dim is None:
            out_dim = hidden_dim * n_heads
        super(GAT, self).__init__(
            in_dim, hidden_dim, n_classes, out_dim=out_dim, 
            **kwargs
        )
        self.n_heads = n_heads 
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
            if self.batch_norm:
                self.batchnorms.append(
                    nn.BatchNorm1d(self.hidden_dim * self.n_heads)
                )
            in_channels = self.hidden_dim * self.n_heads
        self.init_model_state = copy.deepcopy(self.state_dict())


class GatedGCN(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, pos_enc=False, pos_enc_dim=2, 
            **kwargs
        ):
        super(GatedGCN, self).__init__(
            in_dim, hidden_dim, n_classes, pos_enc=pos_enc, 
            pos_enc_dim=pos_enc_dim, **kwargs
        )

        self.dropout = nn.Dropout(self.dropout)
        if self.edge_dim:
            self.embedding_e = nn.Linear(self.edge_dim, self.hidden_dim)

        # Positional Encodings
        self.pos_enc = pos_enc
        self.pos_enc_dim = pos_enc_dim
        if self.pos_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, self.hidden_dim)

        in_channels = self.hidden_dim
        for _ in range(self.n_layers - 1):
            self.convs.append(
                ResGatedGraphConv(
                    in_channels, self.hidden_dim, edge_dim=self.edge_dim
                )
            )
            if self.batch_norm:
                self.batchnorms.append(
                    nn.BatchNorm1d(self.hidden_dim)
                )
            in_channels = self.hidden_dim

        self.convs.append(
            ResGatedGraphConv(
                in_channels, self.out_dim, edge_dim=self.edge_dim
            )
        )
        if self.batch_norm:
            self.batchnorms.append(
                nn.BatchNorm1d(self.out_dim)
            )
        self.init_model_state = copy.deepcopy(self.state_dict())


    def forward(self, x, edge_index, edge_attr=None, x_pos_enc=None):

        x = torch.argmax(x, dim=1)
        x = self.embedding_x(x)
        if self.training:
            x = self.in_feat_dropout(x)

        if self.edge_dim:
            edge_attr = self.embedding_e(edge_attr)

        if self.pos_enc:
            x_pos_enc = self.embedding_pos_enc(x_pos_enc.float())
            x = x + x_pos_enc

        x = self._conv(x, edge_index, edge_attr=edge_attr)
        
        x = self.readout_layer(x)

        return x
    
    def _conv(self, x, edge_index, edge_attr=None):

        for i, conv in enumerate(self.convs):
            x_in = x # for residual connection

            x = conv(x, edge_index, edge_attr=edge_attr)

            if self.batch_norm:
                x = self.batchnorms[i](x)

            if self.activation:
                x = self.activation(x)

            if self.residual:
                x = x_in + x # residual connection
            
            if self.training:
                x = self.dropout(x)

        return x
    

class GPS(BaseModel):
    def __init__(
        self, in_dim, hidden_dim, n_classes, **kwargs
    ):
        raise NotImplementedError()
