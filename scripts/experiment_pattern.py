import argparse
from copy import deepcopy
import json
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from alinc.callbacks import EarlyStoppingCB, BestModelCB
from alinc.constants import NON_ATT_GNNS
from alinc.datasets import load_dataset
from alinc.models import load_model
from alinc.path import EXP_PATH
from alinc.utils import check_gpu, load_optimizer, train, test

def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        prog="Experiment",
        description="Multiple training runs with different models and hyperparameters.",
    )
    parser.add_argument(
        "-o", "--overwrite",
        help="Use this flag to overwrite existing save files.",
        action="store_true"
    )
    parser.add_argument(
        "-pb", "--progressbar",
        help="Use this flag to activate tqdm progress bar",
        action="store_true"
    )
    args, unknown = parser.parse_known_args()

    # Check GPU availability 
    device = check_gpu(log=True)

    # Create experiment directory
    experiment = "pattern/"
    save_dir = os.path.join(EXP_PATH, experiment)

    # Experimental Setting
    dataset = "pattern"
    params = { 
        "batch_size": 64,
        "infer_batch_size": 64,
        "max_epochs": 1000,
        "early_stopping_mode": "max",
        "patience": 20,
        "optimizer": "adam",
        "optimizer_params": {
            "weight_decay": 0.0
        },
        "loss_function": "bcewithlogits",
        "loss_function_params": {},
        "evaluator": "binary_classification",
        "evaluator_params": {},
        "primary_metric": "acc",
        "save_interval": 1,
        "save_model": False,
        "save_optimizer": False,
        "status_interval": 2,
        "num_workers": 0
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        with open(os.path.join(save_dir, "exp_setting.json"), "r") as f:
            existing_params = json.load(f)
        if not(params == existing_params) and not args.overwrite:
            print(
                "Error: Experiment already exists with different setting!" + \
                " Please choose a different experiment name or" +\
                " use '-o' or '--overwrite' to overwrite it."
            )
            sys.exit(1)
    with open(os.path.join(save_dir, "exp_setting.json"), "w") as f:
        json.dump(params, f)

    # Model Parameters
    model_names = ["gat"]
    fix_model_params = {
        "in_dim": 3,
        "n_classes": 2,
        "in_feat_dropout": 0.0,
        "dropout": 0.0,
        "readout": "mean",
        "batch_norm": True,
        "residual": True,
        "device": device
    }

    # Hyperparameters
    hidden_dim = [19]
    n_layers = [4]
    n_heads = [8]
    learning_rate = [0.001]

    # Seeds
    seeds = [100]

    # Load Dataset
    train_dataset = load_dataset(dataset, split="train").to(device)
    val_dataset = load_dataset(dataset, split="val").to(device)
    test_dataset = load_dataset(dataset, split="test").to(device)

    # Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, 
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params["infer_batch_size"], shuffle=False, 
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=params["infer_batch_size"], shuffle=False, 
        drop_last=False,
    )

    # Experiment Loop
    for model_name in model_names:
        hyperparams = {}

        for h_c in hidden_dim:
            hyperparams["hidden_dim"] = h_c

            for n_l in n_layers:
                hyperparams["n_layers"] = n_l

                for lr in learning_rate:
                    hyperparams["learning_rate"] = lr
                
                    for i, h in enumerate(n_heads):
                        if model_name.lower() in NON_ATT_GNNS and i > 0:
                            # Only 1 iteration for non-attentional GNNs
                            continue

                        model_params = fix_model_params | hyperparams
                        if not model_name.lower() in NON_ATT_GNNS:
                            model_params["n_heads"] = h
                        model_save_dir = os.path.join(
                            save_dir, 
                            f"{model_name}" + \
                            f"_hid{model_params['hidden_dim']}" + \
                            f"_nl{model_params['n_layers']}" + \
                            f"_lr{model_params['learning_rate']}" + \
                            f"_h{model_params['n_heads']}/"
                        )
                        if os.path.exists(model_save_dir) and not args.overwrite:
                            print(
                                f"Error: The directory {model_save_dir} already exists." + \
                                " Use '-o' or '--overwrite' to overwrite it." + \
                                " The model configuration will be skipped for now."
                            )
                            continue
                        elif not os.path.exists(model_save_dir):
                            os.makedirs(model_save_dir)
                        
                        for seed in seeds:

                            run_save_dir = os.path.join(model_save_dir, f"seed{seed}")
                            if not os.path.exists(run_save_dir):
                                os.makedirs(run_save_dir)
                            model_params["seed"] = seed
                            torch.manual_seed(seed)

                            with open(os.path.join(run_save_dir, "model_params.json"), "w") as f:
                                json.dump(model_params, f)

                            checkpoint_dir = os.path.join(run_save_dir, "checkpoints/")
                            if not os.path.exists(checkpoint_dir):
                                os.makedirs(checkpoint_dir)

                            print(
                                f"\n{model_name.upper()} (Hidden Channels: {h_c}, " + \
                                f"Num. of Layers: {n_l}, Learning Rate: {lr}, " + \
                                (f"Heads: {model_params['n_heads']}, " if "n_heads" in model_params.keys() else "") + \
                                f"Seed: {seed})\n"
                            )

                            # Load Model
                            model = load_model(model_name, **model_params)
                            model.to(device)
                            optimizer = load_optimizer(
                                params["optimizer"], model.parameters(), 
                                lr=hyperparams["learning_rate"], 
                                **params["optimizer_params"]
                            )
                            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                optimizer, mode=params["early_stopping_mode"], 
                                factor=0.5, patience=5
                            )
                            early_stopping = EarlyStoppingCB(
                                mode=params["early_stopping_mode"], patience=params["patience"]
                            )
                            best_model_cb = BestModelCB(
                                mode=params["early_stopping_mode"]
                            )

                            # Training Loop
                            best_metrics = None
                            best_state_dict = None
                            primary_metric = params["primary_metric"]
                            result_array = None
                            epochs = range(1, params["max_epochs"] + 1)
                            if args.progressbar:
                                epochs = tqdm(epochs)
                            for epoch in epochs:

                                t0 = time.perf_counter()
                                train_metrics, optimizer = train(model, train_loader, optimizer)
                                t1 = time.perf_counter()
                                val_metrics = test(model, val_loader)
                                t2 = time.perf_counter()

                                # Display status
                                message = f"Epoch {epoch}/{params['max_epochs']} - " \
                                    + f"Loss: {train_metrics['loss']:.4f} - " \
                                    + f"Train {primary_metric.upper()}: " \
                                    + f"{train_metrics[primary_metric]:.4f} - " \
                                    + f"Val {primary_metric.upper()}: " \
                                    + f"{val_metrics[primary_metric]:.4f} - " \
                                    + f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
                                
                                if args.progressbar:
                                    epochs.set_description(message)
                                elif epoch % params["status_interval"] == 0:
                                    print(message)

                                # Save Metrics
                                train_metrics_prefix = {"train_" + k: v for k, v in train_metrics.items()}
                                val_metrics_prefix = {"val_" + k: v for k, v in val_metrics.items()}
                                metrics = train_metrics_prefix | val_metrics_prefix
                                metrics["training_time"] = t1 - t0
                                metrics["inference_time"] = t2 - t1
                                if result_array is None:
                                    result_array = np.array(list(metrics.values()))[np.newaxis] 
                                else:
                                    result_array = np.concatenate(
                                        (result_array, np.array(list(metrics.values()))[np.newaxis]), axis=0
                                    )
                                pd.DataFrame(
                                    result_array,
                                    columns = list(metrics.keys())
                                ).to_csv(os.path.join(run_save_dir, "results.csv"))

                                # Save Best Model
                                if best_model_cb(val_metrics[primary_metric]):
                                    if params["save_model"]:
                                        model_checkpoint = os.path.join(run_save_dir, "best_model.pth")
                                        torch.save(model.state_dict(), model_checkpoint)
                                    else:
                                        best_state_dict = deepcopy(model.state_dict())
                                    if params["save_optimizer"]:
                                        opt_checkpoint = os.path.join(run_save_dir, "best_optimizer.pth")
                                        torch.save(optimizer.state_dict(), opt_checkpoint)
                                    best_metrics = metrics

                                # Regular Checkpoint
                                if epoch % params["save_interval"] == 0:
                                    if params["save_model"]:
                                        model_checkpoint = os.path.join(checkpoint_dir, f"model_ckpt{epoch}.pth")
                                        torch.save(model.state_dict(), model_checkpoint)
                                    if params["save_optimizer"]:
                                        opt_checkpoint = os.path.join(checkpoint_dir, f"optimizer_ckpt{epoch}.pth")
                                        torch.save(optimizer.state_dict(), opt_checkpoint)

                                # LR Scheduler
                                lr_scheduler.step(val_metrics[primary_metric])

                                # Early Stopping
                                if early_stopping(val_metrics[primary_metric]):
                                    if not args.progressbar:
                                        print(message)
                                    print(
                                        f"Early Stopping! Best Val {primary_metric}: " \
                                        + f"{early_stopping.best:.4f}"
                                    )
                                    break

                    # Test
                    if params["save_model"]:
                        checkpoint = torch.load(
                            os.path.join(run_save_dir, "best_model.pth"), weights_only=True
                        )
                    else:
                        checkpoint = best_state_dict
                    model.load_state_dict(checkpoint)
                    test_metrics = test(model, test_loader)
                    test_metrics = {"test_" + k : v for k, v in test_metrics.items()}
                    eval_dict = best_metrics | test_metrics
                    with open(os.path.join(run_save_dir, "evaluation.json"), "w") as f:
                        json.dump(eval_dict, f)

if __name__ == "__main__":
    main()