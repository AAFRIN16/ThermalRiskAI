import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import json
from src.dataset import load_config, get_dataloaders
from src.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved at epoch {state['epoch']}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]
    history = checkpoint["history"]
    print(f"Resumed from epoch {checkpoint['epoch']} | Best val acc so far: {best_val_acc:.4f}")
    return start_epoch, best_val_acc, history


def train(cfg):
    device = torch.device(cfg["training"]["device"])
    print(f"Training on: {device}")

    # Data
    train_loader, val_loader, _, _ = get_dataloaders(cfg)

    # Model
    model = build_model(cfg).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"]
    )

    os.makedirs(cfg["outputs"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["outputs"]["logs_dir"], exist_ok=True)

    checkpoint_path = os.path.join(cfg["outputs"]["models_dir"], "checkpoint.pth")
    best_model_path = os.path.join(cfg["outputs"]["models_dir"], "best_model.pth")

    # Resume if checkpoint exists
    start_epoch = 1
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if os.path.exists(checkpoint_path):
        start_epoch, best_val_acc, history = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        patience_counter = 0

    patience = cfg["training"]["early_stopping_patience"]

    print("=" * 50)
    print(f"STARTING TRAINING FROM EPOCH {start_epoch}")
    print("=" * 50)

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_acc: {best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint every epoch
        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
        }, checkpoint_path)

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Save training history
    history_path = os.path.join(cfg["outputs"]["logs_dir"], "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 50)
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"History saved to: {history_path}")
    print("=" * 50)


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)