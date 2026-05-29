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
from dal_toolbox.metrics.utils import entropy_from_logits

class UncertaintySampling(Query, ABC):
    def __init__(self, subset_size=None, random_seed=None, aggr_type="max"):
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
        

    @torch.no_grad()
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
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size
        )
        outputs = model.get_model_outputs(
            unlabeled_dataloader, output_types=['logits']
        )
        scores = self.get_scores(outputs, aggr=True)
        _, indices = scores.topk(acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        if return_utilities:
            return actual_indices, scores
        return actual_indices
    
    def get_scores(self, outputs, aggr=True):
        scores = self.get_utilities(outputs['logits'])
        if aggr:
            scores = self.aggr_layer(scores, outputs['batch'])
        return scores

    @abstractmethod
    def get_utilities(self, logits):
        pass


class EntropySampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 2:
            raise ValueError(f"Input logits tensor must be 2-dimensional, got shape {logits.shape}")
        return entropy_from_logits(logits)


class LeastConfidentSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 2:
            raise ValueError(f"Input logits tensor must be 2-dimensional, got shape {logits.shape}")
        probas = logits.softmax(-1)
        scores = least_confident_score(probas)
        return scores


class MarginSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 2:
            raise ValueError(f"Input logits tensor must be 2-dimensional, got shape {logits.shape}")
        probas = logits.softmax(-1)
        scores = margin_score(probas)
        return scores


def least_confident_score(probas):
    scores, _ = probas.max(dim=-1)
    scores = 1 - scores
    return scores


def margin_score(probas):
    top_probas, _ = torch.topk(probas, k=2, dim=-1)
    scores = top_probas[:, 0] - top_probas[:, 1]
    scores = 1 - scores
    return scores
