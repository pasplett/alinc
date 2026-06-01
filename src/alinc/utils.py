import numpy as np
from omegaconf import OmegaConf
import logging
import os
import random
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from alinc.transforms import (
    VOCNodeNorm, 
    VOCEdgeNorm,
    COCONodeNorm,
    COCOEdgeNorm
)


logger = logging.getLogger(__name__)


def check_gpu(log=True):
    if torch.cuda.is_available():
        if log:
            logger.info("Running on GPU")
        device = "cuda:0"
    else:
        if log:
            logger.info("Running on CPU")
        device = "cpu"
    return device


def load_optimizer(key, *args, **kwargs):
    opt_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }
    return opt_dict[key.lower()](*args, **kwargs)


def train_epoch(
        model, loader, optimizer, evaluator=None, debug_mode=False,
        clip_grad_norm=False, device=torch.device("cpu")
    ):
    model.train()
    epoch_loss = torch.tensor(0.0, device=device)
    if evaluator is not None:
        evaluator.reset()
    for iter, batch in enumerate(loader):

        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(batch)
        loss = model.loss(logits, batch.y)

        epoch_loss += loss.detach()
        if evaluator is not None:
            _logits = logits.detach().to("cpu", non_blocking=True)
            _y = batch.y.detach().to("cpu", non_blocking=True)
            if evaluator.requires_batch:
                _batch = batch.batch.detach().cpu()
                evaluator.update(_logits, _y, _batch)
            else:
                evaluator.update(_logits, _y)

        loss.backward()

        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if debug_mode and iter + 1 >= 5:
            break

    train_stats = evaluator.compute() if evaluator is not None else {}
    train_stats["loss"] = epoch_loss.cpu().item() / (iter + 1)
    return train_stats, optimizer


def test_epoch(
        model, loader, evaluator, debug_mode=False, device=torch.device("cpu")
    ):
    model.eval()
    evaluator.reset()
    for iter, batch in enumerate(loader):

        batch = batch.to(device)
        
        with torch.no_grad():
            logits = model(batch)

        _logits = logits.detach().cpu()
        _y = batch.y.detach().cpu()
        if evaluator.requires_batch:
            _batch = batch.batch.detach().cpu()
            evaluator.update(_logits, _y, _batch)
        else:
            evaluator.update(_logits, _y)

        if debug_mode and iter + 1 >= 5:
            break

    test_stats = evaluator.compute()
    return test_stats


train_epoch.__test__ = False
test_epoch.__test__ = False


def build_datasets(args):
    from alinc.datasets import load_dataset

    transform = None
    if args.dataset.name.lower() == "pascalvoc-sp":
        transform = Compose([VOCNodeNorm(), VOCEdgeNorm()])
    elif args.dataset.name.lower() == "coco-sp":
        transform = Compose([COCONodeNorm(), COCOEdgeNorm()])

    train_ds = load_dataset(args.dataset.name, split="train", transform=transform)
    val_ds = load_dataset(args.dataset.name, split="val", transform=transform)
    test_ds = load_dataset(args.dataset.name, split="test", transform=transform)

    return train_ds, val_ds, test_ds


def build_dataloaders(args):
    train_ds, val_ds, test_ds = build_datasets(args)

    train_loader = DataLoader(
        train_ds, batch_size=args.model.train_batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.model.predict_batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.model.predict_batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0
    )

    return train_loader, val_loader, test_loader


def build_evaluator(args, num_classes):
    from alinc.evaluators import load_evaluator

    evaluator_params = OmegaConf.to_container(args.dataset.evaluator)
    evaluator_name = evaluator_params.pop("name", None)
    evaluator_params["num_classes"] = num_classes
    return load_evaluator(evaluator_name, **evaluator_params)


def build_model(
        args, num_features, num_classes, edge_dim=None, device=torch.device("cpu")
    ):
    from alinc.models import load_model

    model_params = OmegaConf.to_container(args.model)
    model_name = model_params.pop("name")
    for kw in ["num_epochs", "train_batch_size", "predict_batch_size"]:
        model_params.pop(kw)
    model_params["in_dim"] = num_features
    model_params["n_classes"] = num_classes
    model_params["node_encoder"] = args.dataset.node_encoder
    if edge_dim:
        model_params["edge_encoder"] = args.dataset.edge_encoder
    model_params["edge_dim"] = edge_dim
    if model_params["pos_enc"]:
        model_params["pos_enc_params"] = {
            "max_freqs": args.dataset.eigen.max_freqs,
            "raw_norm_type": args.dataset.eigen.norm_type
        }
    model = load_model(model_name, **model_params)
    model.to(device)

    optimizer = load_optimizer(
        args.model.optimizer.name, model.parameters(), 
        lr=args.model.optimizer.lr, 
        weight_decay=args.model.optimizer.weight_decay
    )

    if args.model.lr_scheduler.name == "reduce_lr_on_plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=args.model.early_stopping.mode, 
            factor=args.model.lr_scheduler.reduce_factor, 
            patience=args.model.lr_scheduler.patience
        )
    elif args.model.lr_scheduler.name == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.model.num_epochs
        )
    elif args.model.lr_scheduler.name == "cosine_with_warmup":
        warmup_steps = args.model.lr_scheduler.warmup_steps
        decay_steps = args.model.num_epochs - warmup_steps
        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=args.model.lr_scheduler.start_factor, 
            end_factor=1.0, total_iters=warmup_steps   
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=decay_steps
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        raise NotImplementedError(f"LR Scheduler {args.model.lr_scheduler.name} not known!")

    return model, optimizer, lr_scheduler


def seed_everything(seed: int, strict_mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict_mode:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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
