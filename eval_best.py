import torch
from pathlib import Path

from cifar10_cnn.factory.model_factory import build_model
from cifar10_cnn.utils.datasets import get_loaders
from main import get_best_experiments  # reuse your config

def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    return correct / total

def main():
    # 1) Load best config
    exp_name, cfg = get_best_experiments()[0]   # ("best-sgd", cfg)

    # 2) Build model
    bundle = build_model(cfg)
    model = bundle["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 3) Load trained state dict
    pth_path = Path("results/models/best__best-sgd.pth")
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)

    # 4) Load CIFAR-10 test set
    _, test_loader = get_loaders(
        batch_size=cfg.get("batch_size", 128),
        augment_mode=False,   # ensure NO augmentation on test
        first=None
    )

    # 5) Evaluate
    acc = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
