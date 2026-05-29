import numpy as np
import torch

from torch_geometric.nn.pool import (
    global_max_pool,
    global_mean_pool,
    global_add_pool
)

from dal_toolbox.active_learning.strategies.query import Query
from .centrality import DegreeSampling
from .density import GraphDensitySampling, NodeDensitySampling
from .uncertainty import EntropySampling, LeastConfidentSampling, MarginSampling


class AGE(Query):
    def __init__(
            self, subset_size=None, aggr_type="max",
            uncertainty="entropy", density="node", centrality="degree", 
            alpha=0.25, beta=0.25, gamma=0.5, time_sensitive=False, basef=0.85, 
            aggr_first=True, n_clusters=None
        ):
        super().__init__()
        assert uncertainty in ["entropy", "least_confident", "margin"]
        assert density in ["graph", "node"]
        assert centrality == "degree"

        self.subset_size = subset_size
        self.aggr_type = aggr_type
        self.aggr_first = aggr_first
        self.n_clusters = n_clusters
        if not self.aggr_first:
            if self.aggr_type == "max":
                self.aggr_layer = global_max_pool
            elif self.aggr_type == "mean" or self.aggr_type == "avg":
                self.aggr_layer = global_mean_pool
            elif self.aggr_type == "add" or self.aggr_type == "sum":
                self.aggr_layer = global_add_pool
            else:
                raise KeyError(f"{aggr_type} aggregation does not exist.")
        
        # Uncertainty strategy
        if uncertainty == "entropy":
            self.uncertainty_strategy = EntropySampling(
                subset_size=subset_size, aggr_type=aggr_type
            )
        elif uncertainty == "least_confident":
            self.uncertainty_strategy = LeastConfidentSampling(
                subset_size=subset_size, aggr_type=aggr_type
            )
        elif uncertainty == 'margin':
            self.uncertainty_strategy = MarginSampling(
                subset_size=subset_size, aggr_type=aggr_type
            )
        
        # Density strategy
        if density == "graph":
            assert self.aggr_first
            self.density_strategy = GraphDensitySampling(
                subset_size=subset_size, aggr_type=aggr_type,
                n_clusters=self.n_clusters
            )
        elif density == "node":
            self.density_strategy = NodeDensitySampling(
                subset_size=subset_size, aggr_type=aggr_type,
                n_clusters=self.n_clusters
            )

        # Centrality strategy
        self.centrality_strategy = DegreeSampling(
            subset_size=subset_size, aggr_type=aggr_type
        )

        self.alpha = alpha
        self.beta = beta    
        self.gamma = gamma
        self.time_sensitive = time_sensitive
        self.basef = basef # basef = 0.85 --> distribution approx. uniform at the 20th cycle


    @torch.no_grad()
    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)

        output_types = ['features', 'logits']
        output_types.extend(self.centrality_strategy.output_types)
        outputs = model.get_model_outputs(
            unlabeled_dataloader, output_types=output_types
        )

        if self.n_clusters is None:
            n_clusters = acq_size
        else:
            n_clusters = self.n_clusters

        uncertainty = self.uncertainty_strategy.get_scores(outputs, aggr=self.aggr_first)
        density = self.density_strategy.get_scores(outputs, n_clusters, aggr=self.aggr_first)
        centrality = self.centrality_strategy.get_scores(outputs, aggr=self.aggr_first)
        num_samples = centrality.shape[0]

        # Normalization
        percentile = torch.arange(num_samples, dtype=uncertainty.dtype, device=torch.device("cpu")) / num_samples
        id_sorted = uncertainty.argsort(descending=False)
        uncertainty[id_sorted] = percentile
        id_sorted = density.argsort(descending=False)
        density[id_sorted] = percentile
        id_sorted = centrality.argsort(descending=False)
        centrality[id_sorted] = percentile

        if self.time_sensitive:
            time_value = len(al_datamodule.labeled_indices) / acq_size
            gamma = torch.tensor(np.random.beta(1, 1.005 - self.basef ** time_value))
            alpha = beta = (1 - gamma) / 2
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma

        age_scores = alpha * uncertainty + beta * density + gamma * centrality
        if not self.aggr_first:
            age_scores = self.aggr_layer(age_scores, outputs['batch'])
        _, chosen = torch.topk(age_scores, k=acq_size)
        return [unlabeled_indices[idx] for idx in chosen]
