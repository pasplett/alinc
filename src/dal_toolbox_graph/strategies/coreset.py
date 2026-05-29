import torch
import torch.nn as nn

from torch_geometric.nn.pool import (
    global_max_pool,
    global_mean_pool,
    global_add_pool
)

from ..graph_data import GraphActiveLearningDataModule
from dal_toolbox.active_learning.strategies import CoreSet as OriginalCoreSet


class CoreSet(OriginalCoreSet):
    def __init__(self, subset_size=None, aggr_type="max"):
        super().__init__(subset_size=subset_size)
        self.aggr_type = aggr_type
        if self.aggr_type == "max":
            self.aggr_layer = global_max_pool
        elif self.aggr_type == "mean" or self.aggr_type == "avg":
            self.aggr_layer = global_mean_pool
        elif self.aggr_type == "add" or self.aggr_type == "sum":
            self.aggr_layer = global_add_pool
        else:
            raise KeyError(f"{aggr_type} aggregation does not exist.")
        
    def query(
            self,
            *,
            model: nn.Module,
            al_datamodule: GraphActiveLearningDataModule,
            acq_size: int,
            return_utilities: bool = False,
            # forward_kwargs: dict = None, TODO
            **kwargs
        ):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        labeled_dataloader, _ = al_datamodule.labeled_dataloader()

        unlabeled_outputs = model.get_model_outputs(unlabeled_dataloader, output_types=['features'])
        labeled_outputs = model.get_model_outputs(labeled_dataloader, output_types=['features'])

        features_unlabeled = unlabeled_outputs['features']
        features_labeled = labeled_outputs['features']

        batch_unlabeled = unlabeled_outputs['batch']
        batch_labeled = labeled_outputs['batch']

        features_unlabeled = self.aggr_layer(features_unlabeled, batch_unlabeled)
        features_labeled = self.aggr_layer(features_labeled, batch_labeled)

        chosen = self.kcenter_greedy(features_unlabeled, features_labeled, acq_size)
        return [unlabeled_indices[idx] for idx in chosen]
    

    def kcenter_greedy(self, features_unlabeled: torch.Tensor, features_labeled: torch.Tensor, acq_size: int):
        n_unlabeled = len(features_unlabeled)

        distances = torch.cdist(features_unlabeled, features_labeled)
        min_dist, _ = torch.min(distances, axis=1)

        idxs = []
        for _ in range(acq_size):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = torch.cdist(features_unlabeled, features_unlabeled[idx].unsqueeze(0))
            for j in range(n_unlabeled):
                if j == idx: 
                    min_dist[j] = -1e6 # do not select again
                else:
                    min_dist[j] = torch.min(min_dist[j], dist_new_ctr[j, 0])
        return idxs
