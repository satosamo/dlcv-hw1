from pathlib import Path
import json
import csv
from datetime import datetime

import click
import torch

from cifar10_cnn.factory.model_factory import build_model
from cifar10_cnn.training.train import train_model
from cifar10_cnn.utils.datasets import get_loaders

import numpy as np
import random


RESULTS_DIR = Path("results")
MODE_CHOICES = ["small", "medium", "full"]


# helpers: filesystem + CSV logging

def ensure_result_dirs_existence():
    (RESULTS_DIR/"models").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR/"hists").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR/"plots").mkdir(parents=True, exist_ok=True)


def append_to_csv(task: str, exp_name: str, cfg: dict, hist: dict):
    csv_path = RESULTS_DIR / "experiments.csv"

    # final epoch metrics
    last_idx = len(hist["val_acc"]) - 1
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "exp_name": exp_name,
        "final_epoch": last_idx + 1,
        "final_train_loss": hist["train_loss"][last_idx],
        "final_val_loss": hist["val_loss"][last_idx],
        "final_train_acc": hist["train_acc"][last_idx],
        "final_val_acc": hist["val_acc"][last_idx],
        "config_json": json.dumps(cfg),
    }

    fieldnames = list(row.keys())
    is_new = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def get_data_and_epochs(cfg: dict, size: str):
    # TODO: this is messy
    """
    Decide number of examples, augmentation and epochs from size.
    - small: subset of 1000 images for quick debugging
    - medium: 1/6 of dataset, 15 epochs
    - full: full dataset, 30 epochs
    """
    if size == "small":
        first = 1000
        epochs = cfg.get("epochs_small", 8)
    elif size == "medium":
        first = 10000
        epochs = cfg.get("epochs_medium", 15)
    elif size == "full":
        first = None
        epochs = cfg.get("epochs_full", 100)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    augment = cfg.get("augment_mode", False)
    batch_size = cfg.get("batch_size", 64)
    return first, augment, batch_size, epochs


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# core experiment runner
 
def run_experiment(task: str, exp_name: str, cfg: dict, size: str):
    """
    Build model + optimizer from cfg, run training, save model, hist + CSV row.
    """
    ensure_result_dirs_existence()
    print(f"\n=== [{task}] {exp_name} (size={size}) ===")

    bundle = build_model(cfg)
    model = bundle["model"]
    optimizer = bundle["optimizer"]
    scheduler = bundle["scheduler"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    first, augment_mode, batch_size, epochs = get_data_and_epochs(cfg, size)

    train_loader, val_loader = get_loaders(
        batch_size=batch_size,
        augment_mode=augment_mode,
        first=first,
    )

    hist = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=torch.nn.CrossEntropyLoss(),
        epochs=epochs,
        device=device,
        scheduler=scheduler,
    )

    # save data for tasks
    hist_path = RESULTS_DIR / "hists" / f"{task}__{exp_name}.pt"
    model_path = RESULTS_DIR / "models" / f"{task}__{exp_name}.pth"

    torch.save(hist, hist_path)
    torch.save(model.state_dict(), model_path)

    append_to_csv(task, exp_name, cfg, hist)

    print(f"Saved history to {hist_path}")
    print(f"Saved model state_dict to {model_path}")
    print(f"Final val_acc: {hist['val_acc'][-1]:.3f}")


# Basic model task

# base config
def base_cfg() -> dict:
    """
    Base training config that all experiments start from.
    """
    return {
        "in_channels": 3,
        "hidden_feat_out_channels": [32, 64, 128],
        "hidden_cls_out_dims": [256, 128],
        "output_dim": 10,

        "activation_feat": "relu",
        "activation_cls": "relu",
        "norm_feat": None,
        "norm_cls": None,

        "kernel_size_conv": 3,
        "padding": 1,
        "kernel_size_pool": 2,

        "dropout_feat": 0.0,
        "dropout_cls": 0.0,
        "residual_feat": False,
        "residual_cls": False,

        "optimizer": "adam",
        "lr": 1e-3,

        "batch_size": 256,
        "epochs_small": 50,
        "epochs_medium": 15,
        "epochs_full": 100,

        "augment_mode": None,
    }

# Baseline Task

def get_baseline_experiments():
    exps = []
    cfg = base_cfg()
    cfg["epochs_full"] = 40

    name = "baseline"
    exps.append((name, cfg))

    return exps


# Activations task

def get_activation_experiments():
    """
    For 'activations' task: vary activation function, keep optimizer = Adam.
    """
    acts = ["sigmoid", "tanh", "elu", "leakyrelu03","leakyrelu01","leakyrelu003", "prelu"]

    exps = []
    for act in acts:
        cfg = base_cfg()
        cfg["activation_feat"] = act
        cfg["activation_cls"] = act
        cfg["optimizer"] = "adam"
        cfg["lr"] = 1e-3

        cfg["dropout_feat"] = 0.0
        cfg["dropout"] = 0.0
        cfg["norm_feat"] = None
        cfg["norm_cls"] = None

        cfg["epochs_full"] = 60

        name = f"act-{act}"
        exps.append((name, cfg))

    return exps


# Optimization task

def get_optimizer_experiments():
    exps = []

    # Adam variants
    adam_cfg1 = base_cfg()
    adam_cfg1["optimizer"] = "adam"
    adam_cfg1["lr"] = 0.001
    adam_cfg1["epochs_full"] = 60
    exps.append(("opt-adam-1e-3", adam_cfg1))

    adam_cfg2 = base_cfg()
    adam_cfg2["optimizer"] = "adam"
    adam_cfg2["lr"] = 0.00005
    adam_cfg2["epochs_full"] = 60
    exps.append(("opt-adam-5e-4", adam_cfg2))

    adam_cfg3 = base_cfg()
    adam_cfg3["optimizer"] = "adam"
    adam_cfg3["lr"] = 0.01
    adam_cfg3["epochs_full"] = 60
    exps.append(("opt-adam-1e-2", adam_cfg3))

    adam_cfg4 = base_cfg()
    adam_cfg4["optimizer"] = "adam"
    adam_cfg4["lr"] = 0.03
    adam_cfg4["epochs_full"] = 60
    exps.append(("opt-adam-3e-2", adam_cfg4))

    # SGD variants
    sgd_cfg1 = base_cfg()
    sgd_cfg1["optimizer"] = "sgd"
    sgd_cfg1["lr"] = 0.1
    sgd_cfg1["momentum"] = 0.9
    sgd_cfg1["weight_decay"] = 5e-4
    sgd_cfg1["epochs_full"] = 60
    exps.append(("opt-sgd-lr0.1-m0.9", sgd_cfg1))

    sgd_cfg2 = base_cfg()
    sgd_cfg2["optimizer"] = "sgd"
    sgd_cfg2["lr"] = 0.05
    sgd_cfg2["momentum"] = 0.8
    sgd_cfg2["weight_decay"] = 5e-4
    sgd_cfg2["epochs_full"] = 60
    exps.append(("opt-sgd-lr0.05-m0.8", sgd_cfg2))

    sgd_cfg3 = base_cfg()
    sgd_cfg3["optimizer"] = "sgd"
    sgd_cfg3["lr"] = 0.01
    sgd_cfg3["momentum"] = 0.9
    sgd_cfg3["weight_decay"] = 5e-4
    sgd_cfg3["epochs_full"] = 60
    exps.append(("opt-sgd-lr0.01-m0.9", sgd_cfg3))

    sgd_cfg4 = base_cfg()
    sgd_cfg4["optimizer"] = "sgd"
    sgd_cfg4["lr"] = 0.03
    sgd_cfg4["momentum"] = 0.8
    sgd_cfg4["weight_decay"] = 5e-4
    sgd_cfg4["epochs_full"] = 60
    exps.append(("opt-sgd-lr0.03-m0.8", sgd_cfg4))

    # minibatch test
    # Batch Size 32
    bs32_cfg = base_cfg()
    bs32_cfg["optimizer"] = "sgd"
    bs32_cfg["lr"] = 0.01
    bs32_cfg["batch_size"] = 32
    bs32_cfg["epochs_full"] = 60
    exps.append(("opt-sgd-lr0.01-bs32", bs32_cfg))

    # Batch Size 128
    bs128_cfg = base_cfg()
    bs128_cfg["optimizer"] = "sgd"
    bs128_cfg["lr"] = 0.01
    bs128_cfg["batch_size"] = 128
    bs128_cfg["epochs_full"] = 60
    exps.append(("opt-sgd-lr0.01-bs128", bs128_cfg))
    
    # Batch Size 256
    bs256_cfg = base_cfg()
    bs256_cfg["optimizer"] = "sgd"
    bs256_cfg["lr"] = 0.01
    bs256_cfg["batch_size"] = 256
    bs256_cfg["epochs_full"] = 60
    exps.append(("opt-sgd-lr0.01-bs256", bs256_cfg))

    # RMSprop variant
    rms_cfg = base_cfg()
    rms_cfg["optimizer"] = "rmsprop"
    rms_cfg["lr"] = 0.01
    rms_cfg["alpha"] = 0.99
    rms_cfg["momentum"] = 0.9
    rms_cfg["weight_decay"] = 5e-4
    rms_cfg["epochs_full"] = 60
    exps.append(("opt-rmsprop", rms_cfg))

    return exps


# Dropout and Augmentation task

def get_dropout_experiments():
    """
    For 'dropout' task: test
    - dropout only in FC
    - dropout also after convs
    - different augmentation setups
    - we handle both full train and first=1000 via --size.
    """
    exps = []

    # 1) No dropout, no augmentation (baseline)
    cfg0 = base_cfg()
    cfg0["augment_mode"] = None
    cfg0["dropout_feat"] = 0.0
    cfg0["dropout_cls"] = 0.0
    cfg0["epochs_full"] = 60
    exps.append(("dropout-none-aug-none", cfg0))

    # 2) Dropout only in cls (p = 0.5)
    cfg1 = base_cfg()
    cfg1["augment_mode"] = None
    cfg1["dropout_feat"] = 0.0
    cfg1["dropout_cls"] = 0.5
    cfg1["epochs_full"] = 60
    exps.append(("dropout-fc0.5-aug-none", cfg1))

    # 3) Dropout in feat (p = 0.2) + cls 0.5
    cfg2 = base_cfg()
    cfg2["augment_mode"] = None
    cfg2["dropout_feat"] = 0.2
    cfg2["dropout_cls"] = 0.5
    cfg2["epochs_full"] = 60
    exps.append(("dropout-feat0.2-fc0.5-aug-none", cfg2))

    # 4) Dropout in feat (p = 0.5)
    cfg3 = base_cfg()
    cfg3["augment_mode"] = None
    cfg3["dropout_feat"] = 0.5
    cfg3["dropout_cls"] = 0
    cfg3["epochs_full"] = 60
    exps.append(("dropout-feat0.5-aug-none", cfg3))

    # 5) Augmentation 1: crop + flip
    cfg4 = base_cfg()
    cfg4["augment_mode"] = "basic"
    cfg4["dropout_feat"] = 0.2
    cfg4["dropout_cls"] = 0.5
    cfg4["epochs_full"] = 60
    exps.append(("dropout-fc0.5-aug-basic", cfg4))

    # 6) Augmentation 2: same flag in cfg
    cfg5 = base_cfg()
    cfg5["augment_mode"] = "advanced"
    cfg5["dropout_feat"] = 0.2
    cfg5["dropout_cls"] = 0.5
    cfg5["epochs_full"] = 60
    exps.append(("dropout-feat0.2-fc0.5-aug-advanced", cfg5))

    # 7) Augmentation 2: same flag in cfg
    cfg6 = base_cfg()
    cfg6["augment_mode"] = "best"
    cfg6["dropout_feat"] = 0.2
    cfg6["dropout_cls"] = 0.5
    cfg6["epochs_full"] = 60
    exps.append(("dropout-feat0.2-fc0.5-aug-best", cfg6))

    return exps

def get_dropout_experiments_f1000():
    """
    For 'dropout' task: test
    - dropout only in FC
    - dropout also after convs
    - different augmentation setups
    - we handle both full train and first=1000 via --size.
    """
    exps = []

    # 1) No dropout, no augmentation (baseline)
    cfg0 = base_cfg()
    cfg0["augment_mode"] = None
    cfg0["dropout_feat"] = 0.0
    cfg0["dropout_cls"] = 0.0
    cfg0["epochs_full"] = 50
    exps.append(("dropout-none-aug-none-f1000", cfg0))

    # 2) Dropout only in cls (p = 0.5)
    cfg1 = base_cfg()
    cfg1["augment_mode"] = None
    cfg1["dropout_feat"] = 0.0
    cfg1["dropout_cls"] = 0.5
    cfg1["epochs_full"] = 50
    exps.append(("dropout-fc0.5-aug-none-f1000", cfg1))

    # 3) Dropout in feat (p = 0.2) + cls 0.5
    cfg2 = base_cfg()
    cfg2["augment_mode"] = None
    cfg2["dropout_feat"] = 0.2
    cfg2["dropout_cls"] = 0.5
    cfg2["epochs_full"] = 50
    exps.append(("dropout-feat0.2-fc0.5-aug-none-f1000", cfg2))

    # 4) Dropout in feat (p = 0.5)
    cfg3 = base_cfg()
    cfg3["augment_mode"] = None
    cfg3["dropout_feat"] = 0.5
    cfg3["dropout_cls"] = 0
    cfg3["epochs_full"] = 50
    exps.append(("dropout-feat0.5-aug-none-f1000", cfg3))

    # 5) Augmentation 1: crop + flip
    cfg4 = base_cfg()
    cfg4["augment_mode"] = "basic"
    cfg4["dropout_feat"] = 0.2
    cfg4["dropout_cls"] = 0.5
    cfg4["epochs_full"] = 50
    exps.append(("dropout-fc0.5-aug-basic-f1000", cfg4))

    # 6) Augmentation 2: same flag in cfg
    cfg5 = base_cfg()
    cfg5["augment_mode"] = "advanced"
    cfg5["dropout_feat"] = 0.2
    cfg5["dropout_cls"] = 0.5
    cfg5["epochs_full"] = 50
    exps.append(("dropout-feat0.2-fc0.5-aug-advanced-f1000", cfg5))

    # 7) Augmentation 2: same flag in cfg
    cfg6 = base_cfg()
    cfg6["augment_mode"] = "best"
    cfg6["dropout_feat"] = 0.2
    cfg6["dropout_cls"] = 0.5
    cfg6["epochs_full"] = 50
    exps.append(("dropout-feat0.2-fc0.5-aug-best-f1000", cfg6))

    return exps

# Deep web task

def get_deepnet_experiments():
    """
    Deep web task: build deep (>=11 layers) network + BatchNorm / residual variants.
    """
    exps = []

    deep_channels = [
        32, 32, 32, 
        64, 64, 64, 
        128, 128, 128, 128, 128
    ]

    deep_dims = [256, 128, 64]

    # 1) Deep, no BN, no residuals
    cfg0 = base_cfg()
    cfg0["hidden_feat_out_channels"] = deep_channels
    cfg0["hidden_cls_out_dims"] = deep_dims
    cfg0["norm_feat"] = None
    cfg0["norm_cls"] = None
    cfg0["residual_feat"] = False
    cfg0["residual_cls"] = False
    cfg0["pool_every"] = 3
    cfg0["epochs_full"] = 75
    exps.append(("deep-plain", cfg0))

    # 2) Deep + BatchNorm
    cfg1 = base_cfg()
    cfg1["hidden_feat_out_channels"] = deep_channels
    cfg1["hidden_cls_out_dims"] = deep_dims
    cfg1["norm_feat"] = "batchnorm2d"
    cfg1["norm_cls"] = "batchnorm1d"
    cfg1["residual_feat"] = False
    cfg1["residual_cls"] = False
    cfg1["pool_every"] = 3
    cfg1["epochs_full"] = 75
    exps.append(("deep-bn", cfg1))

    # 3) Deep + BatchNorm + residuals in conv blocks
    cfg2 = base_cfg()
    cfg2["hidden_feat_out_channels"] = deep_channels
    cfg2["hidden_cls_out_dims"] = deep_dims
    cfg2["norm_feat"] = "batchnorm2d"
    cfg2["norm_cls"] = "batchnorm1d"
    cfg2["residual_feat"] = True
    cfg2["residual_cls"] = False
    cfg2["pool_every"] = 3
    cfg2["epochs_full"] = 75
    exps.append(("deep-bn-resfeat", cfg2))

    # 4) Deep + residuals in classifier
    cfg3 = base_cfg()
    cfg3["hidden_feat_out_channels"] = deep_channels
    cfg3["hidden_cls_out_dims"] = deep_dims
    cfg3["norm_feat"] = "batchnorm2d"
    cfg3["norm_cls"] = "layernorm"  # trying layernorm in classifier
    cfg3["residual_feat"] = False
    cfg3["residual_cls"] = True
    cfg3["pool_every"] = 3
    cfg3["epochs_full"] = 75
    exps.append(("deep-bn-ln-rescls", cfg3))

    # 5) Deep + residuals in both
    cfg4 = base_cfg()
    cfg4["hidden_feat_out_channels"] = deep_channels
    cfg4["hidden_cls_out_dims"] = deep_dims
    cfg4["norm_feat"] = "batchnorm2d"
    cfg4["norm_cls"] = "layernorm"
    cfg4["residual_feat"] = True
    cfg4["residual_cls"] = True
    cfg4["pool_every"] = 3
    cfg4["epochs_full"] = 75
    exps.append(("deep-bn-ln-resfeatcls", cfg4))

    return exps


def get_best_experiments():
    cfg0 = base_cfg()

    deep_channels = [
        64, 64, 64,
        128, 128, 128,
        256, 256, 256, 
        512, 512
    ]
    cfg0["hidden_feat_out_channels"] = deep_channels
    cfg0["pool_every"] = 3

    cfg0["residual_feat"] = True
    cfg0["norm_feat"] = "batchnorm2d"
    cfg0["dropout_feat"] = 0.0

    deep_dims = [512]

    cfg0["hidden_cls_out_dims"] = deep_dims
    cfg0["norm_cls"] = "batchnorm1d"
    cfg0["dropout_cls"] = 0.0

    cfg0["optimizer"] = "sgd"
    cfg0["lr"] = 0.1
    cfg0["momentum"] = 0.9 
    cfg0["weight_decay"] = 5e-4
    cfg0["nesterov"] = True

    cfg0["scheduler"] = "cosine" 
    cfg0["t_max"] = 100

    cfg0["augment_mode"] = "best"
    cfg0["batch_size"] = 128
    cfg0["epochs_full"] = 100

    exps = []
    exps.append(("best-sgd", cfg0))

    return exps

# CLI entry point
TASK_MAP = {
    "baseline": get_baseline_experiments,
    "activations": get_activation_experiments,
    "optimizers": get_optimizer_experiments,
    "dropout": get_dropout_experiments,
    "dropoutf1000": get_dropout_experiments_f1000,
    "deep": get_deepnet_experiments,
    "best": get_best_experiments,
}


@click.command()
@click.option(
    "--task",
    type=click.Choice(["baseline", "activations", "optimizers", "dropout", "dropoutf1000", "deep", "best", "all"]),
    default="activations",
    help="Which experiment group to run.",
)
@click.option(
    "--size",
    type=click.Choice(MODE_CHOICES),
    default="small",
    help="small = subset; medium / full = full CIFAR-10.",
)
def cli(task, size):
    if task == "all":
        tasks_to_run = ["baseline", "activations", "optimizers", "dropout", "dropoutf1000", "deep", "best"]
    else:
        tasks_to_run = [task]

    for t in tasks_to_run:
        get_exps = TASK_MAP[t]
        exps = get_exps()
        if t == "dropoutf1000": size = "small"
        for exp_name, cfg in exps:
            set_seed()
            try:
                run_experiment(t, exp_name, cfg, size)
            except Exception as e:
                print(f"!!! FAILED: {exp_name} !!!")
                print(e)
                with open("failures.log", "a") as f:
                    f.write(f"{t} - {exp_name}: {str(e)}\n")
                continue


if __name__ == "__main__":
    cli()
