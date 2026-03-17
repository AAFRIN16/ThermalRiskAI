import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(cfg, mode="train"):
    image_size = cfg["data"]["image_size"]
    mean = cfg["preprocessing"]["normalize_mean"]
    std = cfg["preprocessing"]["normalize_std"]

    if mode == "train" and cfg["preprocessing"]["augmentation"]:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_dataloaders(cfg):
    train_transform = get_transforms(cfg, mode="train")
    val_transform = get_transforms(cfg, mode="val")

    train_dataset = datasets.ImageFolder(
        root=cfg["data"]["train_dir"],
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=cfg["data"]["val_dir"],
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    return train_loader, val_loader, train_dataset, val_dataset


def inspect_dataset(cfg):
    _, _, train_dataset, val_dataset = get_dataloaders(cfg)

    print("=" * 40)
    print("DATASET INSPECTION")
    print("=" * 40)
    print(f"Classes: {train_dataset.classes}")
    print(f"Class-to-index map: {train_dataset.class_to_idx}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset)}")
    print()

    # Count per class
    from collections import Counter
    train_labels = [label for _, label in train_dataset.samples]
    val_labels   = [label for _, label in val_dataset.samples]
    train_counts = Counter(train_labels)
    val_counts   = Counter(val_labels)

    print("Train class distribution:")
    for idx, name in enumerate(train_dataset.classes):
        print(f"  {name}: {train_counts[idx]} images")

    print("Val class distribution:")
    for idx, name in enumerate(val_dataset.classes):
        print(f"  {name}: {val_counts[idx]} images")

    print()
    # Sample one image shape
    sample_img, sample_label = train_dataset[0]
    print(f"Sample image tensor shape: {sample_img.shape}")
    print(f"Sample label: {sample_label} ({train_dataset.classes[sample_label]})")
    print("=" * 40)


if __name__ == "__main__":
    cfg = load_config()
    inspect_dataset(cfg)