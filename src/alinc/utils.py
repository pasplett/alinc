import numpy as np
from omegaconf import OmegaConf
import os
import random
import torch

from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

from alinc.datasets import load_dataset
from alinc.models import load_model


def check_gpu(log=True):
    if torch.cuda.is_available():
        if log:
            print("Running on GPU")
        device = "cuda:0"
    else:
        if log:
            print("Running on CPU")
        device = "cpu"
    return device


def load_optimizer(key, *args, **kwargs):
    opt_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }
    return opt_dict[key.lower()](*args, **kwargs)


def accuracy_SBM(scores, targets):
    """Accuracy for Stochastic Block Model (SBM) datasets 'Pattern' and 
    'Cluster' from https://github.com/graphdeeplearning/benchmarking-gnns
    """
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc


def train_epoch(
        model, loader, optimizer, evaluator, use_edge_attr=False, 
        debug_mode=False
    ):
    model.train()
    epoch_loss = 0.
    epoch_acc_SBM = 0.
    evaluator.reset()
    for iter, g in enumerate(loader):

        x = g.x.to(torch.int64)
        edge_index = g.edge_index
        if use_edge_attr:
            edge_attr = g.edge_attr.to(torch.float32)
        else:
            edge_attr = None
        
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_attr=edge_attr)

        loss = model.loss(logits, g.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_acc_SBM += accuracy_SBM(logits, g.y)
        evaluator.update(logits, g.y)

        if debug_mode:
            break

    train_stats = evaluator.compute()
    train_stats["loss"] = epoch_loss / (iter + 1)
    train_stats["acc_SBM"] = epoch_acc_SBM / (iter + 1)
    return train_stats, optimizer


def test_epoch(model, loader, evaluator, use_edge_attr=False):
    model.eval()
    epoch_acc_SBM = 0.
    evaluator.reset()
    for iter, g in enumerate(loader):

        x = g.x.to(torch.int64)
        edge_index = g.edge_index
        if use_edge_attr:
            edge_attr = g.edge_attr.to(torch.float32)
        else:
            edge_attr = None
        
        logits = model(x, edge_index, edge_attr=edge_attr)
        epoch_acc_SBM += accuracy_SBM(logits, g.y)
        evaluator.update(logits, g.y)

    test_stats = evaluator.compute()
    test_stats["acc_SBM"] = epoch_acc_SBM / (iter + 1)
    return test_stats


def build_datasets(args, device=torch.device("cpu")):

    train_ds = load_dataset(args.dataset, split="train", device=device)
    val_ds = load_dataset(args.dataset, split="val", device=device)
    test_ds = load_dataset(args.dataset, split="test", device=device)
    num_classes = torch.max(train_ds[0].y).item() + 1

    train_loader = DataLoader(
        train_ds, batch_size=args.model.train_batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.model.predict_batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.model.predict_batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader, test_loader, num_classes


def build_model(args, num_features, num_classes, device=torch.device("cpu")):

    model_params = OmegaConf.to_container(args.model)
    model_name = model_params.pop("name")
    for kw in ["num_epochs", "train_batch_size", "predict_batch_size"]:
        model_params.pop(kw)
    model_params["in_dim"] = num_features
    model_params["n_classes"] = num_classes
    model = load_model(model_name, **model_params)
    model.to(device)

    optimizer = load_optimizer(
        args.optimizer.name, model.parameters(), 
        lr=args.optimizer.lr, 
        weight_decay=args.optimizer.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    return model, optimizer, lr_scheduler


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_cfg(cfg, parent_key='', sep='.'):
    """From DAL-Toolbox: https://github.com/dhuseljic/dal-toolbox
    """
    from omegaconf import DictConfig
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
