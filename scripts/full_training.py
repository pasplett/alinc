import copy
import os
import tqdm
import time
import hydra
import mlflow

from omegaconf import OmegaConf

from alinc.callbacks import EarlyStoppingCB, BestModelCB
from alinc.evaluators import load_evaluator
from alinc.utils import (
    build_datasets,
    build_model,
    check_gpu,
    flatten_cfg,
    seed_everything,
    test_epoch,
    train_epoch
)


# os.environ["POSSIBLE_USER_WARNINGS"] = "off"

@hydra.main(version_base=None, config_path="./configs", config_name="full_training")
def main(args):
    seed_everything(42)
    print(OmegaConf.to_yaml(args))

    if args.debug_mode:
        print("CAUTION! DEBUG MODE")

    # GPU
    device = check_gpu(log=True)

    # Dataset
    train_loader, val_loader, test_loader, num_classes = build_datasets(
        args, device=device
    )
    num_features = train_loader.dataset[0].x.shape[1]
    
    # Seed
    seed_everything(args.random_seed)

    # Model
    model, optimizer, lr_scheduler = build_model(
        args, num_features=num_features, num_classes=num_classes, device=device
    )

    # Evaluation
    evaluator = load_evaluator(
        args.evaluator.name, average=args.evaluator.average, 
        num_classes=num_classes, device=device
    )

    # Callbacks
    early_stopping = EarlyStoppingCB(
        mode=args.early_stopping.mode, patience=args.early_stopping.patience
    )
    best_model_cb = BestModelCB(mode=args.early_stopping.mode)

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
            debug_mode=args.debug_mode
        )
        t1 = time.perf_counter()
        val_stats = test_epoch(model, val_loader, evaluator)
        t2 = time.perf_counter()

        # Epoch statistics
        val_performance = val_stats[args.primary_metric]
        train_stats_prefix = {"train_" + k: v for k, v in train_stats.items()}
        val_stats_prefix = {"val_" + k: v for k, v in val_stats.items()}
        stats = train_stats_prefix | val_stats_prefix
        stats["training_time"] = t1 - t0
        stats["inference_time"] = t2 - t1
        history.append(stats)

        # LR scheduler
        lr_scheduler.step(val_performance)

        # Logging
        if args.logging:
            # Display status
            message = f"Epoch {epoch + 1}/{num_epochs} - " \
                + f"Loss: {stats['train_loss']:.4f} - " \
                + f"Train {args.primary_metric.upper()}: "\
                + f"{train_stats[args.primary_metric]:.4f} - " \
                + f"Val {args.primary_metric.upper()}: "\
                + f"{val_performance:.4f} - " \
                + f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
            
            if args.progressbar:
                epochs.set_description(message)
            elif (epoch + 1) % args.status_interval == 0:
                print(message)

        # Save best model
        if best_model_cb(val_performance):
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        # Early stopping
        if early_stopping(val_performance):
            if not args.progressbar:
                print(message)
            print(
                f"Early Stopping! Best Val {args.primary_metric.upper()}: " \
                + f"{early_stopping.best:.4f}"
            )
            break

    # Test set evaluation
    model.load_state_dict(best_state_dict)
    test_stats = test_epoch(model, test_loader, evaluator)
    test_stats["epoch"] = best_epoch
    test_stats = {"test_" + k : v for k, v in test_stats.items()}

    # MLflow
    abs_path = os.path.abspath(args.mlflow_dir)
    tracking_uri = f"sqlite:///{abs_path}/{args.mlflow_db}"
    artifact_location = f"file:///{abs_path}/mlruns"
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
