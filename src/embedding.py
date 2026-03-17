import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml
from src.dataset import load_config


def load_features(cfg):
    features_dir = cfg["outputs"]["features_dir"]
    all_features = np.load(os.path.join(features_dir, "all_features.npy"))
    all_labels = np.load(os.path.join(features_dir, "all_labels.npy"))
    return all_features, all_labels


def compute_umap_embedding(features, n_components=2, n_neighbors=15, min_dist=0.1):
    print("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("Computing UMAP embedding...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=True
    )
    embedding = reducer.fit_transform(features_scaled)
    print(f"Embedding shape: {embedding.shape}")
    return embedding, scaler, reducer


def compute_ndvii(features, labels, embedding):
    """
    Compute a simple NDVII score per sample based on:
    - Distance from Control Group centroid in embedding space
    - Normalized to [0, 1]
    """
    control_mask = labels == 0
    dm_mask = labels == 1

    control_centroid = embedding[control_mask].mean(axis=0)
    dm_centroid = embedding[dm_mask].mean(axis=0)

    # Distance of each point from control centroid
    distances = np.linalg.norm(embedding - control_centroid, axis=1)

    # Normalize to [0, 1]
    ndvii_scores = (distances - distances.min()) / (distances.max() - distances.min())

    print(f"NDVII scores — Control Group mean: {ndvii_scores[control_mask].mean():.4f}")
    print(f"NDVII scores — DM Group mean:      {ndvii_scores[dm_mask].mean():.4f}")

    return ndvii_scores, control_centroid, dm_centroid


def plot_embedding(embedding, labels, ndvii_scores, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    class_names = ["Control Group", "DM Group"]
    colors = ["#00e5ff", "#ff4b6e"]

    # --- Plot 1: Class-colored embedding ---
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1320")
    ax.set_facecolor("#0d1320")

    for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == cls_idx
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=cls_name,
            s=18, alpha=0.75, edgecolors="none"
        )

    ax.set_title("Perfusion Stability Embedding (PSE) — UMAP 2D",
                 color="white", fontsize=13, pad=12)
    ax.set_xlabel("UMAP-1", color="#5a7090")
    ax.set_ylabel("UMAP-2", color="#5a7090")
    ax.tick_params(colors="#5a7090")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d47")

    legend = ax.legend(facecolor="#131b2e", edgecolor="#1e2d47", labelcolor="white")
    plt.tight_layout()
    path1 = os.path.join(save_dir, "pse_embedding_classes.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path1}")

    # --- Plot 2: NDVII score heatmap ---
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1320")
    ax.set_facecolor("#0d1320")

    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=ndvii_scores, cmap="RdYlGn_r",
        s=18, alpha=0.85, edgecolors="none",
        vmin=0, vmax=1
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("NDVII Score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("NDVII Score Distribution in PSE Space",
                 color="white", fontsize=13, pad=12)
    ax.set_xlabel("UMAP-1", color="#5a7090")
    ax.set_ylabel("UMAP-2", color="#5a7090")
    ax.tick_params(colors="#5a7090")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d47")

    plt.tight_layout()
    path2 = os.path.join(save_dir, "pse_ndvii_heatmap.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path2}")


def run_embedding(cfg):
    print("=" * 50)
    print("PHASE 5: PSE EMBEDDING + NDVII")
    print("=" * 50)

    features, labels = load_features(cfg)
    print(f"Loaded features: {features.shape}, labels: {labels.shape}")

    embedding, scaler, reducer = compute_umap_embedding(features)

    ndvii_scores, control_centroid, dm_centroid = compute_ndvii(
        features, labels, embedding
    )

    # Save embedding and scores
    features_dir = cfg["outputs"]["features_dir"]
    np.save(os.path.join(features_dir, "umap_embedding.npy"), embedding)
    np.save(os.path.join(features_dir, "ndvii_scores.npy"), ndvii_scores)

    print(f"Embedding saved to: {features_dir}/umap_embedding.npy")
    print(f"NDVII scores saved to: {features_dir}/ndvii_scores.npy")

    # Plot
    plot_embedding(embedding, labels, ndvii_scores, cfg["outputs"]["features_dir"])

    print("=" * 50)
    print("Phase 5 complete.")
    print("=" * 50)


if __name__ == "__main__":
    cfg = load_config()
    run_embedding(cfg)