import torch
import torcheval.metrics as evalm


def load_evaluator(key, *args, **kwargs):
    evaluator_dict = {
        "binary_classification": BinaryClassificationEvaluator,
        "multiclass_classification": MulticlassClassificationEvaluator,
    }
    return evaluator_dict[key.lower()](*args, **kwargs)


class BaseEvaluator:
    def __init__(self, **kwargs):
        self.metrics = {}
    
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
            **kwargs
        ):
        super().__init__()
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
            self.metrics_avg = {
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
                )
            }
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
