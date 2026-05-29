import copy
import tqdm
import time
import hydra
import mlflow
import logging
import torch
from pathlib import Path

from omegaconf import OmegaConf


mlflow.config.enable_async_logging()
logging.getLogger("sklearn").setLevel(logging.ERROR)

@hydra.main(version_base=None, config_path="./configs", config_name="full_training")
def main(args):
    from alinc.callbacks import EarlyStoppingCB, BestModelCB
    from alinc.utils import (
        build_datasets,
        build_evaluator,
        build_model,
        check_gpu,
        flatten_cfg,
        seed_everything,
        test_epoch,
        train_epoch
    )

    seed_everything(42)
    print(OmegaConf.to_yaml(args))

    if args.debug_mode:
        print("CAUTION! DEBUG MODE")

    # GPU
    device = check_gpu(log=True)

    # Dataset
    train_loader, val_loader, test_loader = build_datasets(args)
    num_features = train_loader.dataset[0].x.shape[1]
    num_classes = torch.max(train_loader.dataset.y).item() + 1

    # Edge attributes
    edge_dim = None
    if not(train_loader.dataset[0].edge_attr is None):
        edge_dim = train_loader.dataset[0].edge_attr.shape[1]
    
    # Seed
    seed_everything(args.random_seed)

    # Model
    model, optimizer, lr_scheduler = build_model(
        args, num_features=num_features, num_classes=num_classes,
        edge_dim=edge_dim, device=device
    )

    # Evaluation
    evaluator = build_evaluator(args, num_classes)

    # Callbacks
    if args.early_stopping:
        early_stopping = EarlyStoppingCB(
            mode=args.model.early_stopping.mode, 
            patience=args.model.early_stopping.patience
        )
        best_model_cb = BestModelCB(mode=args.model.early_stopping.mode)

    # Training loop
    num_epochs = args.model.num_epochs
    epochs = range(num_epochs)
    history = []
    if args.logging and args.progressbar:
        epochs = tqdm.tqdm(epochs)
    for epoch in epochs:

        # Train and evaluate
        t0 = time.perf_counter()
        train_stats, optimizer = train_epoch(
            model, train_loader, optimizer, evaluator, 
            debug_mode=args.debug_mode, device=device,
            clip_grad_norm=args.model.optimizer.clip_grad_norm
        )
        t1 = time.perf_counter()
        val_stats = test_epoch(
            model, val_loader, evaluator, debug_mode=args.debug_mode, 
            device=device
        )
        t2 = time.perf_counter()

        # Epoch statistics
        val_performance = val_stats[args.dataset.primary_metric]
        train_stats_prefix = {"train_" + k: v for k, v in train_stats.items()}
        val_stats_prefix = {"val_" + k: v for k, v in val_stats.items()}
        stats = train_stats_prefix | val_stats_prefix
        stats["training_time"] = t1 - t0
        stats["inference_time"] = t2 - t1
        history.append(stats)

        # LR scheduler
        if args.model.lr_scheduler.name == "reduce_lr_on_plateau":
            lr_scheduler.step(val_performance)
        else:
            lr_scheduler.step()

        # Logging
        if args.logging:
            # Display status
            message = f"Epoch {epoch + 1}/{num_epochs} - " \
                + f"Loss: {stats['train_loss']:.4f} - " \
                + f"Train {args.dataset.primary_metric.upper()}: "\
                + f"{train_stats[args.dataset.primary_metric]:.4f} - " \
                + f"Val {args.dataset.primary_metric.upper()}: "\
                + f"{val_performance:.4f} - " \
                + f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
            
            if args.progressbar:
                epochs.set_description(message)
            elif (epoch + 1) % args.status_interval == 0:
                print(message)

        if args.early_stopping:
            # Save best model
            if best_model_cb(val_performance):
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            # Early stopping
            if early_stopping(val_performance):
                if not args.progressbar:
                    print(message)
                print(
                    f"Early Stopping! Best Val {args.dataset.primary_metric.upper()}: " \
                    + f"{early_stopping.best:.4f}"
                )
                break

    # Test set evaluation
    if args.early_stopping:
        model.load_state_dict(best_state_dict)
    test_stats = test_epoch(
        model, test_loader, evaluator, debug_mode=args.debug_mode, device=device
    )
    test_stats["epoch"] = best_epoch if args.early_stopping else args.model.num_epochs
    test_stats = {"test_" + k : v for k, v in test_stats.items()}

    # MLflow
    mlflow_dir = Path(args.mlflow_dir).resolve()
    tracking_uri = f"sqlite:///{(mlflow_dir / args.mlflow_db).as_posix()}"
    artifact_location = (mlflow_dir / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri=tracking_uri)
    if not mlflow.get_experiment_by_name(args.experiment_name):
        mlflow.create_experiment(
            args.experiment_name, 
            artifact_location=artifact_location
        )
    experiment_id = mlflow.set_experiment(args.experiment_name).experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(flatten_cfg(args))
        for i, stats in enumerate(history):
            mlflow.log_metrics(stats, step=i)
        mlflow.log_dict(test_stats, f'test_stats')
        mlflow.pytorch.log_model(model)


if __name__ == '__main__':
    main()
