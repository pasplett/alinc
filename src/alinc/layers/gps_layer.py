from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj
import torch_geometric.nn as pygnn
from torch_geometric.utils import to_dense_batch

from .gatedgcn_layer import GatedGCNLayer


class GPSLayer(nn.Module):
    """General, powerful, scalable Graph Transformer (GPS) with CustomGatedGCN
    as local MPNN and Transformer for global attention.
    
    Paper:
    https://arxiv.org/pdf/2205.12454

    Code based on: 
    https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py
    """

    def __init__(self, dim_h, num_heads, activation=F.relu, dropout=0.0, 
                 attn_dropout=0.0, layer_norm=False, batch_norm=True, 
                 log_attn_weights=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = activation

        self.log_attn_weights = log_attn_weights

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True

        # MPNNs without edge attributes support.
        self.local_model = GatedGCNLayer(
            dim_h, dim_h, dropout=dropout, residual=True, activation=activation
        )

        # Global attention transformer-style model.
        self.self_attn = torch.nn.MultiheadAttention(
            dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        h = x
        h_in1 = h  # for first residual connection

        h_out_list = []
        h_local, edge_attr = self.local_model(
            x=h, edge_index=edge_index, edge_attr=edge_attr
        )

        if self.layer_norm:
            h_local = self.norm1_local(h_local, batch)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        h_out_list.append(h_local)

        # Multi-head attention.
        h_dense, mask = to_dense_batch(h, batch)
        h_attn = self._sa_block(h_dense, None, ~mask)[mask]

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batch)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch)
        if self.batch_norm:
            h = self.norm2(h)

        return h, edge_attr

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s