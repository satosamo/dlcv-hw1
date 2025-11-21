import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


def train_one_epoch(
    model, 
    loader, 
    optimizer, 
    loss_fn, 
    device
):


    model.train()
    running_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Training", leave=False, mininterval=0.5)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()

        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Validating", leave=False, mininterval=0.5)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def train_model(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    loss_fn, 
    epochs, 
    device, 
    scheduler=None,
    patience=None
):

    hist = {
        "train_loss": [], 
        "val_loss": [], 
        "train_acc": [], 
        "val_acc": []
    }

    if patience is None:
        patience = 30
        print("No patience specifief, setting default value to 15.")

    # early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # training loss, training accuracy, val. loss, val. accuracy
        tl, ta = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        vl, va = evaluate(model, val_loader, loss_fn, device)

        if scheduler:
            scheduler.step()

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_acc={ta:.3f} val_acc={va:.3f}")

        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            best_model_state = model.state_dict() 
            print(f"*** New best model saved (val_loss: {best_val_loss:.4f}) ***")
        else:
            patience_counter += 1
            print(f"*** Validation loss did not improve. Patience: {patience_counter}/{patience} ***")

        if patience_counter >= patience:
            print(f"\n--- Early stopping triggered after {epoch+1} epochs. ---")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return hist
