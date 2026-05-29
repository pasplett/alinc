import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    GATConv,
    GATv2Conv,
    ResGatedGraphConv
)
from torch_geometric.nn.models import MLP

from collections import defaultdict

from alinc.encoders import LapPENodeEncoder
from alinc.layers import GatedGCNLayer, GPSLayer
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def load_model(key, *args, **kwargs):
    model_dict = {
        "gcn": GCN,
        "gin": GIN,
        "gine": GINE,
        "gat": GAT,
        "gatv2": GATv2,
        "gatedgcn_pyg": GatedGCN_PyG,
        "gatedgcn_lrgb": GatedGCN_LRGB,
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
            edge_dim=None, edge_hidden_dim=None, node_encoder="linear", 
            edge_encoder="linear", in_feat_dropout=0.0, dropout=0.0, 
            batch_norm=True, residual=True, activation="relu", self_loops=False, 
            pos_enc=False, pos_enc_dim=2, pos_enc_params={}, readout_type="mlp", 
            n_readout_layers=2, device=torch.device("cpu"), **kwargs
        ):
        super(BaseModel, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if out_dim is None:
            self.out_dim = hidden_dim
        else:
            self.out_dim = out_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.edge_dim = edge_dim
        if edge_hidden_dim is None:
            self.edge_hidden_dim = self.hidden_dim
        else:
            self.edge_hidden_dim = edge_hidden_dim

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        if node_encoder:
            assert node_encoder in ["embedding", "linear", "ogbmol"]
        self.node_encoder = node_encoder
        if edge_encoder:
            assert edge_encoder in ["embedding", "linear", "ogbmol"]
        self.edge_encoder = edge_encoder
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.batch_norm = batch_norm
        self.residual = residual
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise NotImplementedError(f"Activation {activation} not implemented.")
        self.self_loops = self_loops    
        self.device = device

        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        if self.node_encoder == "embedding":
            self.encoder_x = nn.Embedding(self.in_dim, self.hidden_dim)
        elif self.node_encoder == "linear":
            self.encoder_x = nn.Linear(self.in_dim, self.hidden_dim)
        elif self.node_encoder == "ogbmol":
            self.encoder_x = AtomEncoder(self.hidden_dim)

        if self.edge_dim:
            if self.edge_encoder == "embedding":
                self.encoder_e = nn.Embedding(self.edge_dim, self.edge_hidden_dim)
            elif self.edge_encoder == "linear":
                self.encoder_e = nn.Linear(self.edge_dim, self.edge_hidden_dim)
            elif self.edge_encoder == "ogbmol":
                self.encoder_e = BondEncoder(self.edge_hidden_dim)

        self.pos_enc = pos_enc
        self.pos_enc_dim = pos_enc_dim
        self.pos_enc_params = pos_enc_params
        if self.pos_enc:
            if self.node_encoder == "embedding":
                self.encoder_x = nn.Embedding(
                    self.in_dim, self.hidden_dim - self.pos_enc_dim
                )
            elif self.node_encoder == "linear":
                self.encoder_x = nn.Linear(
                    self.in_dim, self.hidden_dim - self.pos_enc_dim
                )
            elif self.node_encoder == "ogbmol":
                self.encoder_x = AtomEncoder(
                    self.hidden_dim - self.pos_enc_dim
                )
            self.encoder_pe = LapPENodeEncoder(
                hidden_dim, dim_pe=self.pos_enc_dim, **self.pos_enc_params
            )

        self.readout_type = readout_type
        self.n_readout_layers = n_readout_layers
        if self.readout_type.lower() == "mlp":
            self.readout_layer = MLPReadout(
                self.out_dim, self.n_classes, n_layers=self.n_readout_layers
            )
        elif self.readout_type.lower() == "linear":
            self.readout_layer = nn.Linear(self.out_dim, self.n_classes)
        else:
            raise NotImplementedError(
                f"Unknown readout type {self.readout_layer}!"
            )

    def reset_state(self):
        self.load_state_dict(self.init_model_state)
        
    def get_device(self):
        return next(self.parameters()).device

    def forward_features(self, batch):
        x, edge_index = batch.x, batch.edge_index
        if self.node_encoder == "embedding":
            x = torch.argmax(x.to(torch.int64), dim=1)

        x = self.encoder_x(x)

        if self.pos_enc:
            pe = self.encoder_pe(batch.EigVals, batch.EigVecs)
            x = torch.cat([x, pe], dim=1)

        if self.training:
            x = self.in_feat_dropout(x)

        edge_attr = batch.edge_attr
        if self.edge_dim and not(edge_attr is None):
            edge_attr = self.encoder_e(edge_attr)

        x = self._conv(x, edge_index, edge_attr=edge_attr)
        return x
    
    def forward_head(self, features):
        return self.readout_layer(features)

    def forward(self, batch):
        features = self.forward_features(batch)
        return self.forward_head(features)
    
    def _conv(self, x, edge_index, edge_attr=None):

        for i, conv in enumerate(self.convs):
            x_in = x # for residual connection

            if self.edge_dim and not(edge_attr is None):
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
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
    
    def loss(self, pred, label):
        """See github.com/graphdeeplearning/benchmarking-gnns
        """
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(label.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
    
    @torch.no_grad()
    def get_model_outputs(
            self, dataloader, output_types, use_precomputed_features=False, 
            **kwargs
        ):
        self.eval()

        device = self.get_device()
        outputs = defaultdict(list)
        n_graphs = 0
        for g_batch, _ in dataloader:

            g_batch = g_batch.to(device)
            
            batch = g_batch.batch + n_graphs
            n_graphs += dataloader.batch_size
            outputs["batch"].append(batch.cpu())

            features = None
            for output_type in output_types:
                if output_type == 'logits':
                    if features is None:
                        features = self.compute_features(
                            g_batch, use_precomputed_features=use_precomputed_features
                        )
                    logits = self.forward_head(features)
                    outputs['logits'].append(logits.cpu())
                    del logits
                elif output_type == 'features':
                    if features is None:
                        features = self.compute_features(
                            g_batch, use_precomputed_features=use_precomputed_features
                        )
                    outputs['features'].append(features.cpu())
                elif output_type == 'labels':
                    labels = g_batch.y
                    outputs['labels'].append(labels.cpu())
                elif output_type == 'degrees':
                    degrees = g_batch.degree
                    outputs['degrees'].append(degrees.cpu())
                elif output_type == 'ppr':
                    ppr = g_batch.ppr.cpu()
                    outputs['ppr'].append(ppr)
                else:
                    raise NotImplementedError()   

            if not(features is None):
                del features 
            torch.cuda.empty_cache()

        outputs = {key: torch.cat(val) if isinstance(
            val[0], torch.Tensor) else val for key, val in outputs.items()}
        return outputs
    
    @torch.no_grad()
    def compute_features(self, g_batch, use_precomputed_features=False):
        if use_precomputed_features and hasattr(g_batch, "features"):
            features = g_batch.features
        else:
            features = self.forward_features(g_batch)
        return features


class GCN(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, **kwargs
        ):
        super(GCN, self).__init__(in_dim, hidden_dim, n_classes, **kwargs)

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
        num_layers = len(self.convs)

        for i, conv in enumerate(self.convs):
            x_in = x # for residual connection
            is_last_layer = (i == num_layers - 1)

            x = conv(x, edge_index)

            if self.batch_norm:
                x = self.batchnorms[i](x)

            if self.activation and not is_last_layer:
                x = self.activation(x)

            if self.residual:
                x = x_in + x # residual connection
            
            if self.training and not is_last_layer:
                x = self.dropout(x)

        return x
    

class GIN(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, train_eps=True, 
            n_layers_mlp_gin=2, **kwargs
        ):
        super(GIN, self).__init__(in_dim, hidden_dim, n_classes, **kwargs)

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

        self.init_model_state = copy.deepcopy(self.state_dict())

    def _conv(self, x, edge_index, edge_attr=None):
        num_layers = len(self.convs)

        for i, conv in enumerate(self.convs):
            x_in = x # for residual connection
            is_last_layer = (i == num_layers - 1)

            x = conv(x, edge_index)

            if self.batch_norm:
                x = self.batchnorms[i](x)

            if self.activation and not is_last_layer:
                x = self.activation(x)

            if self.residual:
                x = x_in + x # residual connection
            
            if self.training and not is_last_layer:
                x = self.dropout(x)

        return x
    

class GINE(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, train_eps=True, 
            n_layers_mlp_gin=2, **kwargs
        ):
        super(GINE, self).__init__(in_dim, hidden_dim, n_classes, **kwargs)

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
                GINEConv(
                    nn=mlp, train_eps=self.train_eps, 
                    edge_dim=self.edge_hidden_dim
                )
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
                GINEConv(nn=mlp, train_eps=self.train_eps, edge_dim=self.edge_hidden_dim)
        )
        if self.batch_norm:
            self.batchnorms.append(
                nn.BatchNorm1d(self.out_dim)
            )

        self.init_model_state = copy.deepcopy(self.state_dict())


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
        if self.node_encoder == "embedding":
            self.encoder_x = nn.Embedding(
                self.in_dim, self.hidden_dim * self.n_heads
            )
        elif self.node_encoder == "linear":
            self.encoder_x = nn.Linear(
                self.in_dim, self.hidden_dim * self.n_heads
            )
        elif self.node_encoder == "ogbmol":
            self.encoder_x = AtomEncoder(
                self.hidden_dim * self.n_heads
            )

        if self.pos_enc:
            if self.node_encoder == "embedding":
                self.encoder_x = nn.Embedding(self.in_dim, self.hidden_dim * self.n_heads - self.pos_enc_dim)
            elif self.node_encoder == "linear":
                self.encoder_x = nn.Linear(self.in_dim, self.hidden_dim * self.n_heads - self.pos_enc_dim)
            elif self.node_encoder == "ogbmol":
                self.encoder_x = AtomEncoder(
                    self.hidden_dim * self.n_heads - self.pos_enc_dim
                )
            self.encoder_pe = LapPENodeEncoder(
                self.hidden_dim * self.n_heads, dim_pe=self.pos_enc_dim,
                **self.pos_enc_params
            )

        in_channels = self.hidden_dim * self.n_heads

        for _ in range(self.n_layers):
            self.convs.append(
                GATConv(
                    in_channels, self.hidden_dim, heads=self.n_heads, 
                    edge_dim=self.edge_hidden_dim, concat=True, dropout=self.dropout_rate,
                    add_self_loops=self.self_loops
                )
            )
            if self.batch_norm:
                self.batchnorms.append(
                    nn.BatchNorm1d(self.hidden_dim * self.n_heads)
                )
            in_channels = self.hidden_dim * self.n_heads
        self.init_model_state = copy.deepcopy(self.state_dict())



class GATv2(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, n_heads=1, **kwargs
        ):
        out_dim = kwargs.pop("out_dim", None)
        if out_dim is None:
            out_dim = hidden_dim * n_heads
        super(GATv2, self).__init__(
            in_dim, hidden_dim, n_classes, out_dim=out_dim, 
            **kwargs
        )
        self.n_heads = n_heads 
        if self.node_encoder == "embedding":
            self.encoder_x = nn.Embedding(
                self.in_dim, self.hidden_dim * self.n_heads
            )
        elif self.node_encoder == "linear":
            self.encoder_x = nn.Linear(
                self.in_dim, self.hidden_dim * self.n_heads
            )
        elif self.node_encoder == "ogbmol":
            self.encoder_x = AtomEncoder(
                self.hidden_dim * self.n_heads
            )

        if self.pos_enc:
            if self.node_encoder == "embedding":
                self.encoder_x = nn.Embedding(self.in_dim, self.hidden_dim * self.n_heads - self.pos_enc_dim)
            elif self.node_encoder == "linear":
                self.encoder_x = nn.Linear(self.in_dim, self.hidden_dim * self.n_heads - self.pos_enc_dim)
            elif self.node_encoder == "ogbmol":
                self.encoder_x = AtomEncoder(
                    self.hidden_dim * self.n_heads - self.pos_enc_dim
                )
            self.encoder_pe = LapPENodeEncoder(
                self.hidden_dim * self.n_heads, dim_pe=self.pos_enc_dim,
                **self.pos_enc_params
            )

        in_channels = self.hidden_dim * self.n_heads

        for _ in range(self.n_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels, self.hidden_dim, heads=self.n_heads, 
                    edge_dim=self.edge_hidden_dim, concat=True, dropout=self.dropout_rate,
                    add_self_loops=self.self_loops
                )
            )
            if self.batch_norm:
                self.batchnorms.append(
                    nn.BatchNorm1d(self.hidden_dim * self.n_heads)
                )
            in_channels = self.hidden_dim * self.n_heads
        self.init_model_state = copy.deepcopy(self.state_dict())


class GatedGCN_PyG(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, **kwargs
        ):
        super(GatedGCN_PyG, self).__init__(
            in_dim, hidden_dim, n_classes, **kwargs
        )

        for _ in range(self.n_layers - 1):
            self.convs.append(
                ResGatedGraphConv(
                    self.hidden_dim, self.hidden_dim, 
                    edge_dim=self.edge_hidden_dim if self.edge_dim else None
                )
            )
            if self.batch_norm:
                self.batchnorms.append(
                    nn.BatchNorm1d(self.hidden_dim)
                )

        self.convs.append(
            ResGatedGraphConv(
                self.hidden_dim, self.out_dim, 
                edge_dim=self.edge_hidden_dim if self.edge_dim else None
            )
        )
        if self.batch_norm:
            self.batchnorms.append(
                nn.BatchNorm1d(self.out_dim)
            )
        self.init_model_state = copy.deepcopy(self.state_dict())
    

class GatedGCN_LRGB(BaseModel):
    def __init__(
            self, in_dim, hidden_dim, n_classes, **kwargs
        ):
        super(GatedGCN_LRGB, self).__init__(
            in_dim, hidden_dim, n_classes, **kwargs
        )

        if self.edge_dim is None:
            self.edge_dim = in_dim
        assert self.edge_encoder == "linear"
        assert self.edge_hidden_dim == self.hidden_dim

        in_channels = self.hidden_dim
        for _ in range(self.n_layers - 1):
            self.convs.append(
                GatedGCNLayer(
                    in_channels, self.hidden_dim, self.dropout_rate, self.residual,
                    activation=self.activation
                )
            )
            in_channels = self.hidden_dim

        self.convs.append(
            GatedGCNLayer(
                in_channels, self.out_dim, 0.0, self.residual,
                activation=nn.Identity()
            )
        )

        self.init_model_state = copy.deepcopy(self.state_dict())


    def forward_features(self, batch):
        x, edge_index = batch.x, batch.edge_index
        if self.node_encoder == "embedding":
            x = torch.argmax(x.to(torch.int64), dim=1)

        x = self.encoder_x(x)

        if self.pos_enc:
            pe = self.encoder_pe(batch.EigVals, batch.EigVecs)
            x = torch.cat([x, pe], dim=1)

        if self.training:
            x = self.in_feat_dropout(x)

        edge_attr = batch.edge_attr
        if edge_attr is None:
            # Dummy edge attributes
            n_edges = edge_index.shape[1]
            edge_attr = torch.ones(n_edges, self.edge_dim, device=x.device)
        edge_attr = self.encoder_e(edge_attr)

        x = self._conv(x, edge_index, edge_attr)
        return x
    

    def _conv(self, x, edge_index, edge_attr):

        for conv in self.convs:
            x, edge_attr = conv(x, edge_index, edge_attr)

        return x
    

class GPS(BaseModel):
    def __init__(
        self, in_dim, hidden_dim, n_classes, n_heads=1, att_dropout=0.5, 
        layer_norm=False, **kwargs
    ):
        super(GPS, self).__init__(in_dim, hidden_dim, n_classes, **kwargs)
            
        assert self.hidden_dim == self.out_dim
        self.n_heads = n_heads
        self.att_dropout = att_dropout
        self.layer_norm = layer_norm
        assert self.layer_norm != self.batch_norm

        if self.edge_dim is None:
            self.edge_dim = in_dim
            assert self.edge_encoder == "linear"
            assert self.edge_hidden_dim == self.hidden_dim
            self.encoder_e = nn.Linear(self.edge_dim, self.hidden_dim)

        for _ in range(self.n_layers - 1):
            conv = GPSLayer(self.hidden_dim, self.n_heads, activation=self.activation,
                            dropout=self.dropout_rate, attn_dropout=self.att_dropout,
                            layer_norm=self.layer_norm, batch_norm=self.batch_norm)
            self.convs.append(conv)

        conv = GPSLayer(self.hidden_dim, self.n_heads, activation=self.activation,
                        dropout=self.dropout_rate, attn_dropout=self.att_dropout, 
                        layer_norm=self.layer_norm, batch_norm=self.batch_norm)
        self.convs.append(conv)

        self.init_model_state = copy.deepcopy(self.state_dict())


    def forward_features(self, batch):
        x, edge_index = batch.x, batch.edge_index
        if self.node_encoder == "embedding":
            x = torch.argmax(x.to(torch.int64), dim=1)

        x = self.encoder_x(x)

        if self.pos_enc:
            pe = self.encoder_pe(batch.EigVals, batch.EigVecs)
            x = torch.cat([x, pe], dim=1)

        if self.training:
            x = self.in_feat_dropout(x)

        edge_attr = batch.edge_attr
        if edge_attr is None:
            # Dummy edge attributes
            n_edges = edge_index.shape[1]
            edge_attr = torch.ones(n_edges, self.edge_dim, device=x.device)
        edge_attr = self.encoder_e(edge_attr)

        x = self._conv(x, edge_index, edge_attr, batch.batch)
        return x
    

    def _conv(self, x, edge_index, edge_attr, batch):

        for conv in self.convs:
            x, edge_attr = conv(x, edge_index, edge_attr=edge_attr, batch=batch)

        return x
