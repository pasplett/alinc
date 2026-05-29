# Implementation of https://arxiv.org/abs/2202.02794.
# Code partially from https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/typiclust.py

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from torch_geometric.nn.pool import (
    global_max_pool,
    global_mean_pool,
    global_add_pool
)

from ..graph_data import GraphActiveLearningDataModule
from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.models.base import BaseModule


def get_nn(features, num_neighbors):
    features = features.numpy().astype(np.float32)
    nn_calculator = NearestNeighbors(n_neighbors=num_neighbors + 1,
                                     metric='sqeuclidean', n_jobs=-1).fit(features)
    distances, indices = nn_calculator.kneighbors(features)

    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    km = KMeans(n_clusters=num_clusters, n_init='auto')
    km.fit_predict(features)
    return km.labels_


class TypiClust(Query):
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, subset_size=None, aggr_type="max", random_seed=None, device='cpu'):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.device = device
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
    def query(self,
              *,
              model: BaseModule,
              al_datamodule: GraphActiveLearningDataModule,
              acq_size: int,
              **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        num_clusters = min(len(labeled_indices) + acq_size, self.MAX_NUM_CLUSTERS)

        unlabeled_outputs = model.get_model_outputs(unlabeled_dataloader, output_types=['features'])
        unlabeled_features = unlabeled_outputs['features']
        unlabeled_batch = unlabeled_outputs['batch']
        unlabeled_features = self.aggr_layer(unlabeled_features, unlabeled_batch)
        if len(labeled_indices) > 0:
            labeled_outputs = model.get_model_outputs(labeled_dataloader, output_types=['features'])
            labeled_features = labeled_outputs['features']
            labeled_batch = labeled_outputs['batch']
            labeled_features = self.aggr_layer(labeled_features, labeled_batch)
        else:
            labeled_features = torch.Tensor([])

        features = torch.cat((labeled_features, unlabeled_features))

        # See https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/pool/_typi_clust.py
        clusters = kmeans(features, num_clusters=num_clusters)
        cluster_sizes = np.zeros(num_clusters)
        cluster_ids, cluster_id_sizes = np.unique(clusters, return_counts=True)
        cluster_sizes[cluster_ids] = cluster_id_sizes
        covered_clusters = np.unique(clusters[:len(labeled_indices)])
        if len(covered_clusters) > 0:
            cluster_sizes[covered_clusters] = 0

        query_indices = []
        for i in range(acq_size):
            if cluster_sizes.max() == 0:
                indices_ = np.arange(len(unlabeled_features))
                indices_ = np.setdiff1d(indices_, query_indices)
                idx = self.rng.choice(indices_)
                query_indices.append(idx)
            else:
                cluster_id = cluster_sizes.argmax()
                cluster_indices = (clusters == cluster_id).nonzero()[0]
                cluster_features = features[cluster_indices]
                typicality = calculate_typicality(cluster_features, min(self.K_NN, len(cluster_indices) // 2))

                idx = typicality.argmax()
                idx = cluster_indices[idx]
                query_indices.append(idx-len(labeled_features))
                cluster_sizes[cluster_id] = 0

        actual_indices = [unlabeled_indices[idx] for idx in query_indices]
        return actual_indices


class InverseTypiClust(TypiClust):
    @torch.no_grad()
    def query(self,
              *,
              model: BaseModule,
              al_datamodule: GraphActiveLearningDataModule,
              acq_size: int,
              **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        num_clusters = min(len(labeled_indices) + acq_size, self.MAX_NUM_CLUSTERS)

        unlabeled_outputs = model.get_model_outputs(unlabeled_dataloader, output_types=['features'])
        unlabeled_features = unlabeled_outputs['features']
        unlabeled_batch = unlabeled_outputs['batch']
        unlabeled_features = self.aggr_layer(unlabeled_features, unlabeled_batch)
        if len(labeled_indices) > 0:
            labeled_outputs = model.get_model_outputs(labeled_dataloader, output_types=['features'])
            labeled_features = labeled_outputs['features']
            labeled_batch = labeled_outputs['batch']
            labeled_features = self.aggr_layer(labeled_features, labeled_batch)
        else:
            labeled_features = torch.Tensor([])
        features = torch.cat((labeled_features, unlabeled_features))

        # See https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/pool/_typi_clust.py
        clusters = kmeans(features, num_clusters=num_clusters)
        cluster_sizes = np.zeros(num_clusters)
        cluster_ids, cluster_id_sizes = np.unique(clusters, return_counts=True)
        cluster_sizes[cluster_ids] = cluster_id_sizes
        covered_clusters = np.unique(clusters[:len(labeled_indices)])
        if len(covered_clusters) > 0:
            cluster_sizes[covered_clusters] = 0

        query_indices = []
        for i in range(acq_size):
            if cluster_sizes.min() == np.inf:
                indices_ = np.arange(len(unlabeled_features))
                indices_ = np.setdiff1d(indices_, query_indices)
                idx = self.rng.choice(indices_)
                query_indices.append(idx)
            else:
                cluster_id = cluster_sizes.argmin()
                cluster_indices = (clusters == cluster_id).nonzero()[0]
                cluster_features = features[cluster_indices]
                typicality = calculate_typicality(cluster_features, min(self.K_NN, len(cluster_indices) // 2))

                idx = typicality.argmin()
                idx = cluster_indices[idx]
                query_indices.append(idx-len(labeled_features))
                cluster_sizes[cluster_id] = np.inf

        actual_indices = [unlabeled_indices[idx] for idx in query_indices]
        return actual_indices
