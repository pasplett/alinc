from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("omegaconf")
pytest.importorskip("ogb")

from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from alinc.models import MLPReadout, load_model
from alinc.utils import (
    build_evaluator,
    build_model,
    flatten_cfg,
    load_optimizer,
    seed_everything,
    test_epoch,
    train_epoch,
)


def make_graph(label=0):
    return Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        y=torch.tensor([label, 1, label], dtype=torch.long),
    )


class SimpleEvaluator:
    requires_batch = False

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, y):
        self.correct += int((logits.argmax(dim=1) == y).sum())
        self.total += y.numel()

    def compute(self):
        return {"acc": 100.0 * self.correct / self.total}


class TinyNodeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, batch):
        return self.linear(batch.x)

    def loss(self, logits, y):
        return nn.functional.cross_entropy(logits, y)


def test_mlp_readout_and_gcn_forward_loss_and_reset_state():
    model = load_model(
        "gcn",
        in_dim=2,
        hidden_dim=4,
        n_classes=2,
        n_layers=2,
        batch_norm=False,
        residual=False,
        readout_type="linear",
    )
    graph = make_graph()

    logits = model(graph)
    loss = model.loss(logits, graph.y)
    with torch.no_grad():
        first_param = next(model.parameters())
        original = first_param.clone()
        first_param.add_(1.0)

    model.reset_state()

    assert logits.shape == (3, 2)
    assert loss.ndim == 0
    assert torch.allclose(next(model.parameters()), original)
    assert MLPReadout(4, 2, n_layers=1)(torch.ones(3, 4)).shape == (3, 2)


def test_train_and_test_epoch_run_on_synthetic_node_batches():
    loader = DataLoader([make_graph(0) for _ in range(6)], batch_size=1)
    model = TinyNodeClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    evaluator = SimpleEvaluator()

    train_stats, returned_optimizer = train_epoch(
        model, loader, optimizer, evaluator, debug_mode=True
    )
    test_stats = test_epoch(model, loader, evaluator, debug_mode=True)

    assert returned_optimizer is optimizer
    assert set(train_stats) == {"acc", "loss"}
    assert 0.0 <= train_stats["acc"] <= 100.0
    assert train_stats["loss"] >= 0.0
    assert set(test_stats) == {"acc"}


def test_build_evaluator_build_model_optimizer_and_config_helpers():
    args = OmegaConf.create(
        {
            "dataset": {
                "evaluator": {
                    "name": "sp",
                    "average": "macro",
                    "average_only": True,
                },
                "node_encoder": "linear",
                "edge_encoder": "linear",
                "eigen": {"max_freqs": 4, "norm_type": "none"},
            },
            "model": {
                "name": "gcn",
                "hidden_dim": 4,
                "n_layers": 1,
                "batch_norm": False,
                "residual": False,
                "pos_enc": False,
                "readout_type": "linear",
                "num_epochs": 3,
                "train_batch_size": 2,
                "predict_batch_size": 2,
                "optimizer": {"name": "adam", "lr": 0.01, "weight_decay": 0.0},
                "lr_scheduler": {"name": "cosine"},
            },
        }
    )

    evaluator = build_evaluator(args, num_classes=2)
    model, optimizer, scheduler = build_model(args, 2, 2)
    flat = flatten_cfg(args)

    assert set(evaluator.metrics) == {"acc_macro", "prec_macro", "rec_macro", "f1_macro"}
    assert isinstance(load_optimizer("adamw", model.parameters(), lr=0.01), torch.optim.AdamW)
    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert flat["model.name"] == "gcn"


def test_seed_everything_makes_torch_randomness_repeatable():
    seed_everything(123)
    first = torch.rand(3)
    seed_everything(123)
    second = torch.rand(3)

    assert torch.equal(first, second)


def test_load_model_rejects_unknown_model_names():
    with pytest.raises(KeyError):
        load_model("missing", in_dim=2, hidden_dim=4, n_classes=2)
