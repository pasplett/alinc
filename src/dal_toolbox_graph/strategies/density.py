import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from torch_geometric.nn.pool import (
    global_max_pool,
    global_mean_pool,
    global_add_pool
)
from sklearn.cluster import KMeans

from ..graph_data import GraphActiveLearningDataModule
from dal_toolbox.active_learning.strategies.query import Query


class DensitySampling(Query, ABC):
    def __init__(self, subset_size=None, aggr_type="max", n_clusters=None):
        super().__init__()
        self.subset_size = subset_size
        self.aggr_type = aggr_type
        if self.aggr_type == "max":
            self.aggr_layer = global_max_pool
        elif self.aggr_type == "mean" or self.aggr_type == "avg":
            self.aggr_layer = global_mean_pool
        elif self.aggr_type == "add" or self.aggr_type == "sum":
            self.aggr_layer = global_add_pool
        else:
            raise KeyError(f"{aggr_type} aggregation does not exist.")
        
        self.n_clusters = n_clusters

    @torch.no_grad()
    def query(
            self,
            *,
            model: nn.Module,
            al_datamodule: GraphActiveLearningDataModule,
            acq_size: int,
            return_utilities: bool = False,
            **kwargs
        ):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)

        if self.n_clusters is None:
            n_clusters = acq_size
        else:
            n_clusters = self.n_clusters

        outputs = model.get_model_outputs(
            unlabeled_dataloader, output_types=['features']
        )
        densities = self.get_scores(outputs, n_clusters, aggr=True)
        _, chosen = torch.topk(densities, k=acq_size)
        return [unlabeled_indices[idx] for idx in chosen]
    
    @abstractmethod
    def get_scores(self, outputs, n_clusters, aggr=True):
        pass


class GraphDensitySampling(DensitySampling):
    def get_scores(self, outputs, n_clusters, aggr=True):
        if not aggr:
            raise ValueError("GraphDensitySampling scores cannot be calculated without aggregation!")
        graph_features = self.aggr_layer(
            outputs["features"], outputs["batch"]
        )
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(graph_features)
        centers = kmeans.cluster_centers_
        label = kmeans.predict(graph_features)
        centers = centers[label]
        dist_map = torch.linalg.norm(graph_features - centers, dim=1)
        densities = 1 / (1 + dist_map)
        return densities.to(torch.float32)
    

class NodeDensitySampling(DensitySampling):
    def get_scores(self, outputs, n_clusters, aggr=True):
        node_features = outputs["features"]
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(node_features)
        centers = kmeans.cluster_centers_
        label = kmeans.predict(node_features)
        centers = centers[label]
        dist_map = torch.linalg.norm(node_features - centers, dim=1)
        densities = 1 / (1 + dist_map)
        if aggr:
            densities = self.aggr_layer(densities, outputs["batch"])
        return densities.to(torch.float32)
