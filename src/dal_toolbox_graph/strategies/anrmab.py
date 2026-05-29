import numpy as np
import torch

from torch_geometric.nn.pool import (
    global_add_pool
)

from dal_toolbox.active_learning.strategies.query import Query
from .centrality import DegreeSampling
from .density import GraphDensitySampling, NodeDensitySampling
from .uncertainty import EntropySampling, LeastConfidentSampling, MarginSampling


class ANRMAB(Query):
    def __init__(
            self, subset_size=None, aggr_type="max",
            uncertainty="entropy", density="node", centrality="degree",
            min_probability_strategy=0.2, num_acq=20, acq_size=5, 
            aggr_first=True, n_clusters=None
        ):
        super().__init__()
        self.reset()
        assert uncertainty in ["entropy", "least_confident", "margin"]
        assert density in ["graph", "node"]
        assert centrality == "degree"

        self.subset_size = subset_size
        self.aggr_type = aggr_type
        self.aggr_first = aggr_first
        self.n_clusters = n_clusters
        if not self.aggr_first:
            assert self.aggr_type in ["add", "sum"]
            self.aggr_layer = global_add_pool
        
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

        self.min_probability_strategy = min_probability_strategy
        self.budget = num_acq * acq_size


    def reset(self):
        self.weights = torch.ones(3).to(torch.float32)
        self.reward_terms = []


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
        num_samples = uncertainty.shape[0]

        # Normalization
        percentile = torch.arange(num_samples, dtype=uncertainty.dtype, device=torch.device("cpu")) / num_samples
        id_sorted = uncertainty.argsort(descending=False)
        uncertainty[id_sorted] = percentile
        id_sorted = density.argsort(descending=False)
        density[id_sorted] = percentile
        id_sorted = centrality.argsort(descending=False)
        centrality[id_sorted] = percentile

        query_matrix = torch.stack((density, uncertainty, centrality))
        query_matrix /= query_matrix.sum(1, keepdim=True)
        probabilities = self.weights * (1 - 3 * self.min_probability_strategy) / self.weights.sum() + self.min_probability_strategy 
        phi = probabilities @ query_matrix
        if not self.aggr_first:
            phi = self.aggr_layer(phi, outputs['batch'])
        assert torch.allclose(torch.tensor(1.0), phi.sum())

        chosen = np.random.choice(phi.size(0), size=acq_size, p=phi.numpy(), replace=False)
        self.update(chosen, phi, probabilities, query_matrix, al_datamodule)

        return [unlabeled_indices[idx] for idx in chosen]

    
    def update(self, sampled_indices, phi, probabilities, query_matrix, al_datamodule):

        n_unlabeled = len(al_datamodule.unlabeled_indices)
        n_total = n_unlabeled + len(al_datamodule.labeled_indices)
        for sampled_idx in sampled_indices:

            self.reward_terms.append(1 / (phi[sampled_idx] * n_unlabeled))
            reward = 1 / self.budget * sum(self.reward_terms)

            r_hat = reward * query_matrix[:, sampled_idx] / phi[sampled_idx]
            self.weights *= torch.exp(
                self.min_probability_strategy / 2 * (r_hat + (1 / probabilities) * np.sqrt(np.log(n_total / 0.1) / (3 * self.budget)))
            )
