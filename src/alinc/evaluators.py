import torch
import torcheval.metrics as evalm
from torchmetrics import MatthewsCorrCoef
from torch.nn.utils.rnn import pad_sequence


def load_evaluator(key, *args, **kwargs):
    if key.lower() == "binary_classification":
        return BinaryClassificationEvaluator(*args, **kwargs)
    elif key.lower() == "multiclass_classification":
        return MulticlassClassificationEvaluator(*args, **kwargs)
    elif key.lower() == "sp":
        return SPEvaluator(*args, **kwargs)
    elif key.lower() == "zaretzki":
        return ZaretzkiEvaluator(*args, **kwargs)
    else:
        raise KeyError(f"Evaluator '{key}' not found.")


class BaseEvaluator:
    def __init__(self, **kwargs):
        self.metrics = {}
        self.requires_batch = False
    
    def __len__(self):
        return len(self.metrics)

    def update(self, y_pred, y_true):
        for _, metric in self.metrics.items():
            metric.update(y_pred, y_true)

    def compute(self):
        stats = {}
        for key, metric in self.metrics.items():
            stats[key] = metric.compute()
        return stats
    
    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset()


class BinaryClassificationEvaluator(BaseEvaluator):

    def __init__(self, device=torch.device("cpu"), **kwargs):
        super().__init__()
        self.metrics = {
            "acc": evalm.BinaryAccuracy(device=device),
            "prec": evalm.BinaryPrecision(device=device),
            "rec": evalm.BinaryRecall(device=device),
            "f1": evalm.BinaryF1Score(device=device),
            "auroc": evalm.BinaryAUROC(device=device),
            "auprc": evalm.BinaryAUPRC(device=device)
        }

    def compute(self):
        stats = {}
        for key, metric in self.metrics.items():
            stats[key] = 100 * metric.compute()
        return stats
    

class MulticlassClassificationEvaluator(BaseEvaluator):

    def __init__(
            self, average="macro", num_classes=3, device=torch.device("cpu"),
            average_only=False, **kwargs
        ):
        super().__init__()
        if not average_only:
            self.metrics = {
                "acc": evalm.MulticlassAccuracy(
                    average=None, num_classes=num_classes, device=device
                ),
                "prec": evalm.MulticlassPrecision(
                    average=None, num_classes=num_classes, device=device
                ),
                "rec": evalm.MulticlassRecall(
                    average=None, num_classes=num_classes, device=device
                ),
                "f1": evalm.MulticlassF1Score(
                    average=None, num_classes=num_classes, device=device
                ),
                "auroc": evalm.MulticlassAUROC(
                    average=None, num_classes=num_classes, device=device
                ),
                "auprc": evalm.MulticlassAUPRC(
                    average=None, num_classes=num_classes, device=device
                )
            }
        if not(average is None):
            metrics_avg = {
                f"acc_{average}": evalm.MulticlassAccuracy(
                    average=average, num_classes=num_classes, device=device
                ),
                f"prec_{average}": evalm.MulticlassPrecision(
                    average=average, num_classes=num_classes, device=device
                ),
                f"rec_{average}": evalm.MulticlassRecall(
                    average=average, num_classes=num_classes, device=device
                ),
                f"f1_{average}": evalm.MulticlassF1Score(
                    average=average, num_classes=num_classes, device=device
                ),
                f"auroc_{average}": evalm.MulticlassAUROC(
                    average=average, num_classes=num_classes, device=device
                ),
                f"auprc_{average}": evalm.MulticlassAUPRC(
                    average=average, num_classes=num_classes, device=device
                ),
            }
            if average_only:
                self.metrics = metrics_avg
            else:
                self.metrics =  self.metrics | metrics_avg

        self.num_classes = num_classes
        self.average = average

    
    def compute(self):
        stats = {}
        for key, metric in self.metrics.items():
            metric_stats = metric.compute()
            if not(self.average is None) and self.average in key:
                stats[key] = 100 * metric_stats.item()
            else:
                for c in range(self.num_classes):
                    stats[f"{key}_{c}"] = 100 * metric_stats[c].item()

        return stats


class SPEvaluator(BaseEvaluator):
    """Multiclass evaluator for large superpixel datasets without AUROC and 
    AUPRC computation.
    """

    def __init__(
            self, average="macro", num_classes=3, device=torch.device("cpu"),
            average_only=False, **kwargs
        ):
        super().__init__()
        if not average_only:
            self.metrics = {
                "acc": evalm.MulticlassAccuracy(
                    average=None, num_classes=num_classes, device=device
                ),
                "prec": evalm.MulticlassPrecision(
                    average=None, num_classes=num_classes, device=device
                ),
                "rec": evalm.MulticlassRecall(
                    average=None, num_classes=num_classes, device=device
                ),
                "f1": evalm.MulticlassF1Score(
                    average=None, num_classes=num_classes, device=device
                )
            }
        if not(average is None):
            metrics_avg = {
                f"acc_{average}": evalm.MulticlassAccuracy(
                    average=average, num_classes=num_classes, device=device
                ),
                f"prec_{average}": evalm.MulticlassPrecision(
                    average=average, num_classes=num_classes, device=device
                ),
                f"rec_{average}": evalm.MulticlassRecall(
                    average=average, num_classes=num_classes, device=device
                ),
                f"f1_{average}": evalm.MulticlassF1Score(
                    average=average, num_classes=num_classes, device=device
                )
            }
            if average_only:
                self.metrics = metrics_avg
            else:
                self.metrics =  self.metrics | metrics_avg

        self.num_classes = num_classes
        self.average = average

    
    def compute(self):
        stats = {}
        for key, metric in self.metrics.items():
            metric_stats = metric.compute()
            if not(self.average is None) and self.average in key:
                stats[key] = 100 * metric_stats.item()
            else:
                for c in range(self.num_classes):
                    stats[f"{key}_{c}"] = 100 * metric_stats[c].item()

        return stats
    

class ZaretzkiEvaluator(MulticlassClassificationEvaluator):
    def __init__(
            self, average="macro", num_classes=3, device=torch.device("cpu"),
            average_only=False, **kwargs
        ):
        super().__init__(
            average=average, num_classes=num_classes, device=device, 
            average_only=average_only, **kwargs
        )
        self.metrics["mcc"] = MatthewsCorrCoef(
            task="multiclass", num_classes=self.num_classes
        )
        self.metrics["top2"] = evalm.TopKMultilabelAccuracy(
            criteria="overlap", k=2
        )
        self.metrics["top3"] = evalm.TopKMultilabelAccuracy(
            criteria="overlap", k=3
        )
        self.requires_batch = True

    def update(self, y_pred, y_true, batch):

        # Prepare topk update
        y_pred_topk = y_pred[:, 1]
        y_pred_topk = [y_pred_topk[batch == i] for i in range(torch.max(batch) + 1)]
        y_pred_topk = pad_sequence(y_pred_topk, batch_first=True, padding_value=-1e6)
        y_true_topk = [y_true[batch == i] for i in range(torch.max(batch) + 1)]
        y_true_topk = pad_sequence(y_true_topk, batch_first=True, padding_value=-1e6)

        for key, metric in self.metrics.items():
            if key in ["top2", "top3"]:
                metric.update(y_pred_topk, y_true_topk)
            else:
                metric.update(y_pred, y_true)

    def compute(self):
        stats = {}
        for key, metric in self.metrics.items():
            metric_stats = metric.compute()
            if not(self.average is None) and self.average in key:
                stats[key] = 100 * metric_stats.item()
            elif key == "mcc":
                stats[key] = metric_stats.item()
            elif key in ["top2", "top3"]:
                stats[key] = 100 * metric_stats.item()
            else:
                for c in range(self.num_classes):
                    stats[f"{key}_{c}"] = 100 * metric_stats[c].item()

        return stats