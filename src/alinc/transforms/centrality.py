import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    degree,
    to_networkx
)

from networkx.algorithms.link_analysis import pagerank

class AddNodeDegree(BaseTransform):
    """Adds node degree."""
    def __init__(
            self, 
            attr_name: str = "degree"
        ):
        super().__init__()
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        node_degrees = degree(data.edge_index[0], num_nodes)
        setattr(data, self.attr_name, node_degrees)

        return data
    

class AddPageRank(BaseTransform):
    """Adds (approximate) PageRank centrality using Andersen PPR."""
    def __init__(
            self, 
            alpha: float = 0.2, 
            eps: float = 1e-5,
            attr_name: str = "ppr"
        ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        ppr_scores = pagerank(
            to_networkx(data), alpha=self.alpha, tol=self.eps
        )
        ppr_scores = torch.Tensor(
            np.fromiter(ppr_scores.values(), dtype=float)
        )
        
        setattr(data, self.attr_name, ppr_scores)
        return data