import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool import (
    global_max_pool,
    global_mean_pool,
    global_add_pool
)

from ..graph_data import GraphActiveLearningDataModule
from dal_toolbox.active_learning.strategies import Badge as OriginalBadge
# from rich.progress import track
from sklearn.metrics import pairwise_distances


class Badge(OriginalBadge):
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
        # grad_embedding = model.get_grad_representations(unlabeled_dataloader, device=self.device)

        outputs = model.get_model_outputs(unlabeled_dataloader, output_types=['features', 'logits'])
        features = outputs['features']
        logits = outputs['logits']

        probas = logits.softmax(-1)
        max_indices = probas.argmax(-1)
        num_classes = logits.size(-1)

        factor = F.one_hot(max_indices, num_classes=num_classes) - probas
        grad_embedding = (factor[:, :, None] * features[:, None, :]).flatten(-2)
        grad_embedding = self.aggr_layer(grad_embedding, outputs['batch'])
        
        chosen = kmeans_plusplus(grad_embedding.numpy(), acq_size, rng=self.rng)

        return [unlabeled_indices[idx] for idx in chosen]


def kmeans_plusplus(X, n_clusters, rng):
    # Start with highest grad norm since it is the "most uncertain"
    grad_norm = np.linalg.norm(X, ord=2, axis=1)
    idx = np.argmax(grad_norm)

    all_distances = pairwise_distances(X, X)

    indices = [idx]
    centers = [X[idx]]
    dist_mat = []
    for _ in range(1, n_clusters):
        # Compute the distance of the last center to all samples
        # dist = np.sqrt(np.sum((X - centers[-1])**2, axis=-1))
        dist = all_distances[indices[-1]]

        dist_mat.append(dist)
        # Get the distance of each sample to its closest center
        min_dist = np.min(dist_mat, axis=0)
        min_dist_squared = min_dist**2
        if np.all(min_dist_squared == 0):
            raise ValueError('All distances to the centers are zero!')
        # sample idx with probability proportional to the squared distance
        p = min_dist_squared / np.sum(min_dist_squared)
        if np.any(p[indices] != 0):
            print('Already sampled centers have probability', p)
        idx = rng.choice(range(len(X)), p=p.squeeze())
        indices.append(idx)
        centers.append(X[idx])
    return indices
