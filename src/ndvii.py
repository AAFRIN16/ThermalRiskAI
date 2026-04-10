import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from src.dataset import load_config


def classify_ndvii(score):
    if score < 0.30:
        return "Thermally Stable"
    elif score < 0.55:
        return "Mild Instability"
    elif score < 0.75:
        return "Moderate Instability"
    else:
        return "High Instability"


def evaluate_ndvii(ndvii_scores, labels):
    cfg = load_config()
    print("=" * 50)
    print("NDVII EVALUATION")
    print("=" * 50)

    class_names = ["Control Group", "DM Group"]
    ndvii_preds = (ndvii_scores >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(labels, ndvii_preds, target_names=class_names))

    cm = confusion_matrix(labels, ndvii_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d1320")
    ax.set_facecolor("#0d1320")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        annot_kws={"color": "white", "size": 13}
    )

    ax.set_title("NDVII Confusion Matrix", color="white", pad=12)
    ax.set_xlabel("Predicted", color="#5a7090")
    ax.set_ylabel("Actual", color="#5a7090")
    ax.tick_params(colors="white")

    plt.tight_layout()
    save_path = os.path.join(cfg["outputs"]["features_dir"], "ndvii_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    control_scores = ndvii_scores[labels == 0]
    dm_scores = ndvii_scores[labels == 1]

    print("=" * 50)
    print("NDVII Summary:")
    print(f"  Control Group — mean: {control_scores.mean():.4f}  std: {control_scores.std():.4f}")
    print(f"  DM Group      — mean: {dm_scores.mean():.4f}  std: {dm_scores.std():.4f}")
    print(f"  Separation gap: {dm_scores.mean() - control_scores.mean():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    cfg = load_config()
    from src.embedding import load_features
    features, labels = load_features(cfg)
    ndvii_scores = np.load(os.path.join(cfg["outputs"]["features_dir"], "ndvii_scores.npy"))
    evaluate_ndvii(ndvii_scores, labels)