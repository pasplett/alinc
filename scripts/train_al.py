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

@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    from alinc.utils import (
        build_evaluator,
        build_datasets,
        build_model,
        check_gpu,
        flatten_cfg,
        seed_everything,
        test_epoch,
    )
    from dal_toolbox_graph.graph_data import GraphActiveLearningDataModule

    seed_everything(42)
    print(OmegaConf.to_yaml(args))

    if args.debug_mode:
        print("CAUTION! DEBUG MODE")

    # GPU
    device = check_gpu(log=True)

    # Dataset
    train_ds, val_ds, test_ds = build_datasets(args)
    num_features = train_ds.x.shape[1]
    num_classes = torch.max(train_ds.y).item() + 1
    print(f"Num Classes: ", num_classes)

    # Edge attributes
    edge_dim = None
    if not(train_ds[0].edge_attr is None):
        edge_dim = train_ds.edge_attr.shape[1]

    # Seed
    seed_everything(args.random_seed, strict_mode=args.strict_det_gpu)

    # Model
    model, optimizer, lr_scheduler = build_model(
        args, num_features=num_features, num_classes=num_classes,
        edge_dim=edge_dim, device=device
    )
    init_optimizer_state = copy.deepcopy(optimizer.state_dict())
    init_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())
    model.init_model_state = copy.deepcopy(model.state_dict())

    # AL
    al_datamodule = GraphActiveLearningDataModule(
        train_dataset=train_ds,
        query_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    args.dataset.num_init = args.dataset.acq_size if args.dataset.num_init is None else args.dataset.num_init
    al_datamodule.random_init(n_samples=args.dataset.num_init)

    # Evaluation
    evaluator = build_evaluator(args, num_classes)
    
    al_strategy = build_al_strategy(args, num_features=num_features)

    al_history = []
    artifacts_history = []
    for i_acq in range(0, args.dataset.num_acq+1):
        if i_acq != 0:
            stime = time.time()
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.dataset.acq_size,
            )
            etime = time.time()
            al_datamodule.update_annotations(indices)

        artifacts = {
            'query_indices': indices if i_acq != 0 else al_datamodule.labeled_indices,
            # 'model': model.state_dict(),
        }

        # Reset
        model.reset_state()
        optimizer.load_state_dict(init_optimizer_state)
        lr_scheduler.load_state_dict(init_scheduler_state)

        # Loaders
        train_loader = al_datamodule.train_dataloader()
        val_loader = al_datamodule.val_dataloader()
        test_loader = al_datamodule.test_dataloader()
        
        model, val_stats = train(
            args, model, train_loader, optimizer, lr_scheduler, 
            val_loader=val_loader, evaluator=evaluator,
            device=device
        )
        test_stats = test_epoch(
            model, test_loader, evaluator,
            debug_mode=args.debug_mode, device=device
        )
        test_stats = {"test_" + k : v for k, v in test_stats.items()}
        test_stats = val_stats | test_stats
        test_stats['query_time'] = etime - stime if i_acq != 0 else 0

        print(f'Cycle {i_acq}:', {k: round(v, 3) for k, v in test_stats.items()}, flush=True)
        al_history.append(test_stats)
        artifacts_history.append(artifacts)

    # MLflow
    # tracking_uri = f"sqlite:///{args.mlflow_db}" # for local tests
    # artifact_location = "file://mlruns" # for local tests
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
        for i_acq, test_stats in enumerate(al_history):
            mlflow.log_metrics(test_stats, step=i_acq)
            artifacts = artifacts_history[i_acq]
            mlflow.log_dict(artifacts_history[i_acq], f'artifacts_cycle{i_acq:02d}')


def train(args, model, train_loader, optimizer, lr_scheduler, val_loader=None, 
          evaluator=None, device=torch.device("cpu")):
    from alinc.callbacks import BestModelCB, EarlyStoppingCB
    from alinc.utils import test_epoch, train_epoch

    
    if args.early_stopping:
        assert not(val_loader is None)
        assert not(evaluator is None)
        # Callbacks
        early_stopping = EarlyStoppingCB(
            mode=args.dataset.early_stopping.mode, 
            patience=args.dataset.early_stopping.patience
        )
        best_model_cb = BestModelCB(mode=args.dataset.early_stopping.mode)
        best_state_dict = None
        best_val_stats = None

    # Training loop
    num_epochs = args.model.num_epochs
    epochs = range(num_epochs)
    if args.logging and args.progressbar:
        epochs = tqdm.tqdm(epochs)
    for epoch in epochs:

        # Train and evaluate
        train_stats, optimizer = train_epoch(
            model, train_loader, optimizer, debug_mode=args.debug_mode,
            clip_grad_norm=args.model.optimizer.clip_grad_norm, device=device
        )
        if args.early_stopping or args.model.lr_scheduler.name == "reduce_lr_on_plateau":
            val_stats = test_epoch(
                model, val_loader, evaluator, debug_mode=args.debug_mode,
                device=device
            )
            val_performance = val_stats[args.dataset.primary_metric]
            
        if args.model.lr_scheduler.name == "reduce_lr_on_plateau":
            lr_scheduler.step(val_performance)
        else:
            lr_scheduler.step()

        # Logging
        if args.logging:
            # Display status
            message = f"Epoch {epoch + 1}/{num_epochs} - " \
                + f"Loss: {train_stats['loss']:.4f} - "
            if args.early_stopping:
                message += f"Val {args.dataset.primary_metric.upper()}: "\
                    + f"{val_performance:.4f} - "
                
            message += f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
            
            if args.progressbar:
                epochs.set_description(message)
            elif (epoch + 1) % args.status_interval == 0:
                print(message)
        
        if args.early_stopping:
            # Save best model
            if best_model_cb(val_performance):
                best_state_dict = copy.deepcopy(model.state_dict())
                best_val_stats = val_stats

            # Early stopping
            if early_stopping(val_performance):
                if not args.progressbar:
                    print(message)
                print(
                    f"Early Stopping! Best Val {args.dataset.primary_metric.upper()}: " \
                    + f"{early_stopping.best:.4f}"
                )
                model.load_state_dict(best_state_dict)
                break
    
    if args.early_stopping:
        return model, best_val_stats
    elif not(val_loader is None) and not(evaluator is None):
        val_stats = test_epoch(
            model, val_loader, evaluator, debug_mode=args.debug_mode,
            device=device
        )
        val_stats = {"val_" + k : v for k, v in val_stats.items()}
        return model, val_stats
    else:
        return model


def build_al_strategy(args, num_features=None):
    from dal_toolbox.active_learning import strategies
    from dal_toolbox_graph import strategies as graph_strategies

    subset_size = args.dataset.subset_size
    aggr_type = args.al.aggr_type
    device = args.al.device
    if args.al.strategy == 'random':
        al_strategy = strategies.RandomSampling()
    elif args.al.strategy == "entropy":
        al_strategy = graph_strategies.EntropySampling(
            subset_size=subset_size, aggr_type=aggr_type
        )
    elif args.al.strategy == "least_confident":
        al_strategy = graph_strategies.LeastConfidentSampling(
            subset_size=subset_size, aggr_type=aggr_type
        )
    elif args.al.strategy == 'margin':
        al_strategy = graph_strategies.MarginSampling(
            subset_size=subset_size, aggr_type=aggr_type
        )
    elif args.al.strategy == 'graph_density':
        al_strategy = graph_strategies.GraphDensitySampling(
            subset_size=subset_size, aggr_type=aggr_type,
            n_clusters=args.al.n_clusters
        )
    elif args.al.strategy == 'node_density':
        al_strategy = graph_strategies.NodeDensitySampling(
            subset_size=subset_size, aggr_type=aggr_type,
            n_clusters=args.al.n_clusters
        )
    elif args.al.strategy == 'degree':
        al_strategy = graph_strategies.DegreeSampling(
            subset_size=subset_size, aggr_type=aggr_type
        )
    elif args.al.strategy == "age":
        al_strategy = graph_strategies.AGE(
            subset_size=subset_size,
            aggr_type=aggr_type,
            uncertainty=args.al.params.uncertainty,
            density=args.al.params.density,
            centrality=args.al.params.centrality,
            alpha=args.al.params.alpha,
            beta=args.al.params.beta,
            gamma=args.al.params.gamma,
            time_sensitive=args.al.params.time_sensitive,
            basef=args.al.params.basef,
            aggr_first=args.al.params.aggr_first,
            n_clusters=args.al.params.n_clusters
        )
    elif args.al.strategy == "anrmab":
        al_strategy = graph_strategies.ANRMAB(
            subset_size=subset_size,
            aggr_type=aggr_type,
            uncertainty=args.al.params.uncertainty,
            density=args.al.params.density,
            centrality=args.al.params.centrality,
            min_probability_strategy=args.al.params.min_probability_strategy,
            num_acq=args.dataset.num_acq,
            acq_size=args.dataset.acq_size,
            aggr_first=args.al.params.aggr_first,
            n_clusters=args.al.params.n_clusters
        )
    elif args.al.strategy == "badge":
        al_strategy = graph_strategies.Badge(
            subset_size=subset_size,
            aggr_type=aggr_type,
        )
    elif args.al.strategy == "coreset":
        al_strategy = graph_strategies.CoreSet(
            subset_size=subset_size,
            aggr_type=aggr_type,
        )
    elif args.al.strategy == "typiclust":
        al_strategy = graph_strategies.TypiClust(
            subset_size=subset_size,
            aggr_type=aggr_type,
            random_seed=args.random_seed
        )
    else:
        raise NotImplementedError()
    return al_strategy


if __name__ == '__main__':
    main()
