from pathlib import Path

import torch
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
HISTS_DIR = RESULTS_DIR / "hists"
PLOTS_DIR = RESULTS_DIR / "plots"


def ensure_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_single_history(hist_path: Path):
    """
    Load one history .pt file and create two plots:
    - accuracy vs epoch
    - loss vs epoch
    """
    ensure_plots_dir()
    hist = torch.load(hist_path, map_location="cpu")

    train_acc = hist["train_acc"]
    val_acc = hist["val_acc"]
    train_loss = hist["train_loss"]
    val_loss = hist["val_loss"]
    epochs = range(1, len(train_acc) + 1)

    stem = hist_path.stem  # e.g. "activations__act-relu"

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(stem + " - Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    acc_path = PLOTS_DIR / f"{stem}_acc.png"
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(stem + " - Loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    loss_path = PLOTS_DIR / f"{stem}_loss.png"
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    print(f"Saved plots for {stem} to {acc_path} and {loss_path}")


def main():
    if not HISTS_DIR.exists():
        print(f"No histories found in {HISTS_DIR}")
        return

    hist_files = sorted(HISTS_DIR.glob("*.pt"))
    if not hist_files:
        print(f"No .pt files found in {HISTS_DIR}")
        return

    for hp in hist_files:
        plot_single_history(hp)


if __name__ == "__main__":
    main()
