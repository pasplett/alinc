import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
    GatedGCN layer (Residual Gated Graph ConvNets)
    
    Paper: 
    https://arxiv.org/pdf/1711.07553.pdf

    Code based on:
    https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gatedgcn_layer.py
    """
    def __init__(self, in_dim, out_dim, dropout, residual, activation=F.relu, **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.e = None

    def forward(self, x, edge_index, edge_attr):

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            edge_attr_in = edge_attr

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(edge_attr)
        Dx = self.D(x)
        Ex = self.E(x)

        x, edge_attr = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=edge_attr, Ax=Ax)

        x = self.bn_node_x(x)
        edge_attr = self.bn_edge_e(edge_attr)

        x = self.activation(x)
        edge_attr = self.activation(edge_attr)

        x = F.dropout(x, self.dropout, training=self.training)
        edge_attr = F.dropout(edge_attr, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            edge_attr = edge_attr_in + edge_attr

        return x, edge_attr

    def message(self, Dx_i, Ex_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out
