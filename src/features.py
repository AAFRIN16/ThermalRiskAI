import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
from src.dataset import load_config, get_dataloaders
from src.model import build_model


def extract_features(cfg):
    device = torch.device(cfg["training"]["device"])
    print(f"Extracting features on: {device}")

    # Load data
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(cfg)

    # Load trained model
    model = build_model(cfg).to(device)
    best_model_path = os.path.join(cfg["outputs"]["models_dir"], "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {best_model_path}")

    os.makedirs(cfg["outputs"]["features_dir"], exist_ok=True)

    def extract(loader, dataset, split_name):
        all_features = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader, desc=f"Extracting {split_name}")):
                images = images.to(device)
                _, features = model(images, return_features=True)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Save
        np.save(os.path.join(cfg["outputs"]["features_dir"], f"{split_name}_features.npy"), all_features)
        np.save(os.path.join(cfg["outputs"]["features_dir"], f"{split_name}_labels.npy"), all_labels)

        print(f"{split_name} features shape: {all_features.shape}")
        print(f"{split_name} labels shape:   {all_labels.shape}")
        return all_features, all_labels

    train_features, train_labels = extract(train_loader, train_dataset, "train")
    val_features, val_labels = extract(val_loader, val_dataset, "val")

    # Combine all features for embedding
    all_features = np.concatenate([train_features, val_features], axis=0)
    all_labels = np.concatenate([train_labels, val_labels], axis=0)

    np.save(os.path.join(cfg["outputs"]["features_dir"], "all_features.npy"), all_features)
    np.save(os.path.join(cfg["outputs"]["features_dir"], "all_labels.npy"), all_labels)

    print("=" * 50)
    print(f"All features shape: {all_features.shape}")
    print(f"All labels shape:   {all_labels.shape}")
    print(f"Saved to: {cfg['outputs']['features_dir']}")
    print("=" * 50)

    return all_features, all_labels


if __name__ == "__main__":
    cfg = load_config()
    extract_features(cfg)