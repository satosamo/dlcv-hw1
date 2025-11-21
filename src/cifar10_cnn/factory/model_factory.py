from cifar10_cnn.modules.module_factory import Cifar10_CNN
import torch.optim as optim


def build_model(cfg):
    """
    Factory function: builds model, optimizer, and learning rate scheduler from a config dictionary.
    """

    model = Cifar10_CNN(cfg)

    optimizer_name = cfg.get("optimizer", "adam").lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 0.0)
        )

    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.get("lr", 0.1),
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0),
            nesterov=cfg.get("nesterov", False),
        )

    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=cfg.get("lr", 0.1),
            alpha=cfg.get("alpha", 0.99),
            eps=cfg.get("alpha", 1e-08),
            weight_decay=cfg.get("weight_decay", 0.0),
            momentum=cfg.get("momentum", 0),
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


    scheduler_type = cfg.get("scheduler", None)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("t_max", 100)
        )

    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get("step_size", 30),
            gamma=cfg.get("gamma", 0.1)
        )

    else:
        scheduler = None

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
