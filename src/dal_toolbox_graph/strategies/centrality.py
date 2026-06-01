import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from torch_geometric.nn.pool import (
    global_max_pool,
    global_mean_pool,
    global_add_pool
)

from ..graph_data import GraphActiveLearningDataModule
from dal_toolbox.active_learning.strategies.query import Query

class CentralitySampling(Query, ABC):
    def __init__(
            self, subset_size=None, random_seed=None, aggr_type="max", 
            output_types=None
        ):
        super().__init__(random_seed=random_seed)
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
        self.output_types = output_types or []

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
        outputs = model.get_model_outputs(
            unlabeled_dataloader, output_types=self.output_types
        )
        scores = self.get_scores(outputs, aggr=True)
        _, chosen = torch.topk(scores, k=acq_size)
        return [unlabeled_indices[idx] for idx in chosen]
    
    @abstractmethod
    def get_scores(self, outputs, aggr=True):
        raise NotImplementedError
    

class DegreeSampling(CentralitySampling):
    def __init__(self, subset_size=None, random_seed=None, aggr_type="max"):
        super().__init__(
            subset_size=subset_size, random_seed=random_seed, 
            aggr_type=aggr_type, output_types=['degrees']
        )
    
    def get_scores(self, outputs, aggr=True):
        scores = outputs['degrees']
        if aggr:
            scores = self.aggr_layer(scores, outputs['batch'])
        return scores
    
