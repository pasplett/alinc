import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from dal_toolbox_graph.graph_data import GraphActiveLearningDataModule
from dal_toolbox_graph.strategies import (
    AGE,
    ANRMAB,
    DegreeSampling,
    EntropySampling,
    LeastConfidentSampling,
    MarginSampling,
)
import dal_toolbox_graph.strategies as graph_strategies
from dal_toolbox_graph.strategies.density import GraphDensitySampling
from dal_toolbox_graph.strategies.uncertainty import (
    least_confident_score,
    margin_score,
)
from dal_toolbox_graph.utils import (
    flatten_cfg,
    test_epoch as run_test_epoch,
    train_epoch as run_train_epoch,
)
from torch_geometric.loader import DataLoader


def make_graph_dataset(n=4):
    graphs = []
    for i in range(n):
        graphs.append(
            Data(
                x=torch.ones(2, 1) * (i + 1),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                y=torch.tensor(i % 2),
                degree=torch.tensor([float(i), float(i + 1)]),
            )
        )
    return graphs


class OutputModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def get_model_outputs(self, dataloader, output_types, **kwargs):
        logits, degrees, features, batch = [], [], [], []
        offset = 0
        for graph_batch, _ in dataloader:
            n_nodes = graph_batch.x.size(0)
            node_ids = torch.arange(offset, offset + n_nodes, dtype=torch.float32)
            if "logits" in output_types:
                logits.append(torch.stack([node_ids, -node_ids], dim=1))
            if "degrees" in output_types:
                degrees.append(graph_batch.degree.float())
            if "features" in output_types:
                features.append(torch.stack([node_ids, node_ids + 1], dim=1))
            batch.append(graph_batch.batch + offset // 2)
            offset += n_nodes

        outputs = {"batch": torch.cat(batch)}
        if logits:
            outputs["logits"] = torch.cat(logits)
        if degrees:
            outputs["degrees"] = torch.cat(degrees)
        if features:
            outputs["features"] = torch.cat(features)
        return outputs


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward_features(self, batch):
        return batch.x

    def forward_head(self, features):
        return self.linear(features).mean(dim=0, keepdim=True)

    def loss(self, logits, y):
        return torch.nn.functional.cross_entropy(logits, y.view(-1))


class RecordingEvaluator:
    requires_batch = False

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_updates = 0

    def update(self, logits, y):
        self.num_updates += 1

    def compute(self):
        return {"updates": self.num_updates}


def test_uncertainty_score_helpers():
    probas = torch.tensor([[0.9, 0.1], [0.55, 0.45]])

    assert torch.allclose(least_confident_score(probas), torch.tensor([0.1, 0.45]))
    assert torch.allclose(margin_score(probas), torch.tensor([0.2, 0.9]))


@pytest.mark.parametrize(
    "strategy_cls",
    [EntropySampling, LeastConfidentSampling, MarginSampling, DegreeSampling],
)
def test_query_strategies_return_original_unlabeled_indices(strategy_cls):
    dm = GraphActiveLearningDataModule(make_graph_dataset(), predict_batch_size=2)
    dm.update_annotations([0])

    selected = strategy_cls(aggr_type="max").query(
        model=OutputModel(), al_datamodule=dm, acq_size=2
    )

    assert len(selected) == 2
    assert set(selected).issubset(set(dm.unlabeled_indices))


def test_density_strategy_scores_graph_level_features():
    outputs = {
        "features": torch.tensor([[0.0], [0.1], [10.0], [10.1]]),
        "batch": torch.tensor([0, 0, 1, 1]),
    }

    scores = GraphDensitySampling(aggr_type="mean").get_scores(
        outputs, n_clusters=2, aggr=True
    )

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_removed_pagerank_and_topk_paths_are_not_available():
    assert not hasattr(graph_strategies, "PageRankSampling")

    with pytest.raises(KeyError):
        EntropySampling(aggr_type="topk")


def test_age_and_anrmab_require_degree_centrality():
    AGE(centrality="degree")
    ANRMAB(centrality="degree")

    with pytest.raises(AssertionError):
        AGE(centrality="pagerank")
    with pytest.raises(AssertionError):
        ANRMAB(centrality="pagerank")


def test_flatten_cfg_flattens_nested_mappings():
    cfg = {"dataset": {"name": "pattern"}, "seed": 1}

    assert flatten_cfg(cfg) == {"dataset.name": "pattern", "seed": 1}


def test_train_and_test_epoch_use_model_hooks():
    dataset = make_graph_dataset(2)
    loader = GraphActiveLearningDataModule(
        dataset, train_batch_size=1, predict_batch_size=1
    ).custom_dataloader([0], train=True)
    model = TinyTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_stats, returned_optimizer = run_train_epoch(
        args=None, model=model, loader=loader, optimizer=optimizer, debug_mode=True
    )
    eval_stats = run_test_epoch(
        args=None,
        model=model,
        loader=DataLoader([dataset[0]], batch_size=1),
        evaluator=RecordingEvaluator(),
        debug_mode=True,
    )

    assert returned_optimizer is optimizer
    assert train_stats["loss"] >= 0
    assert eval_stats == {"updates": 1}
