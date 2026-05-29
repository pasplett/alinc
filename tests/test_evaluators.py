import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torcheval")
pytest.importorskip("torchmetrics")

from alinc.evaluators import BaseEvaluator, load_evaluator


class CountingMetric:
    def __init__(self):
        self.total = 0
        self.reset_calls = 0

    def update(self, y_pred, y_true):
        self.total += int((y_pred.argmax(dim=1) == y_true).sum())

    def compute(self):
        return torch.tensor(float(self.total))

    def reset(self):
        self.reset_calls += 1
        self.total = 0


def test_base_evaluator_updates_computes_and_resets_metrics():
    evaluator = BaseEvaluator()
    metric = CountingMetric()
    evaluator.metrics["correct"] = metric

    evaluator.update(torch.tensor([[0.2, 0.8], [0.9, 0.1]]), torch.tensor([1, 1]))

    assert len(evaluator) == 1
    assert evaluator.compute()["correct"].item() == 1.0
    evaluator.reset()
    assert metric.reset_calls == 1
    assert metric.total == 0


def test_load_evaluator_builds_sp_evaluator_without_auc_metrics():
    evaluator = load_evaluator(
        "sp", num_classes=3, average="macro", average_only=True
    )

    evaluator.update(
        torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]),
        torch.tensor([0, 1, 2]),
    )
    stats = evaluator.compute()

    assert set(stats) == {"acc_macro", "prec_macro", "rec_macro", "f1_macro"}
    assert all(value == pytest.approx(100.0) for value in stats.values())


def test_zaretzki_evaluator_handles_batchwise_topk_metrics():
    evaluator = load_evaluator(
        "zaretzki", num_classes=2, average="macro", average_only=True
    )

    evaluator.update(
        torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]]),
        torch.tensor([1, 0, 1, 0]),
        torch.tensor([0, 0, 1, 1]),
    )
    stats = evaluator.compute()

    assert evaluator.requires_batch is True
    assert "mcc" in stats
    assert "top2" in stats
    assert "top3" in stats


def test_load_evaluator_rejects_unknown_names():
    with pytest.raises(KeyError, match="Evaluator 'missing' not found"):
        load_evaluator("missing")

