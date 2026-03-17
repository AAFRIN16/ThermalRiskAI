import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import yaml
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

from src.model import build_model
from src.dataset import load_config
from src.ndvii import classify_ndvii


# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThermalRiskAI",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Mono', monospace;
        background-color: #070a10;
        color: #c8d8f0;
    }
    .main { background-color: #070a10; }
    .block-container { padding-top: 2rem; }

    h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #edf3ff !important; }

    .metric-card {
        background: #0d1320;
        border: 1px solid #1e2d47;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #edf3ff;
    }
    .metric-label {
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #5a7090;
        margin-top: 4px;
    }
    .ndvii-stable { color: #b8ff57 !important; }
    .ndvii-mild { color: #ffb300 !important; }
    .ndvii-moderate { color: #ff9800 !important; }
    .ndvii-high { color: #ff4b6e !important; }

    .disclaimer {
        background: #0d1320;
        border: 1px solid #ffb300;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.75rem;
        color: #ffb300;
        margin-top: 1rem;
    }
    .stButton > button {
        background: rgba(0,229,255,0.1);
        border: 1px solid #00e5ff;
        color: #00e5ff;
        font-family: 'Space Mono', monospace;
        letter-spacing: 1px;
        border-radius: 6px;
        padding: 8px 20px;
        width: 100%;
    }
    .stButton > button:hover {
        background: rgba(0,229,255,0.2);
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Data ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    cfg = load_config()
    device = torch.device("cpu")  # Dashboard always runs on CPU

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(
        os.path.join(cfg["outputs"]["models_dir"], "best_model.pth"),
        map_location=device
    ))
    model.eval()

    # Load precomputed embeddings
    features_dir = cfg["outputs"]["features_dir"]
    all_features = np.load(os.path.join(features_dir, "all_features.npy"))
    all_labels = np.load(os.path.join(features_dir, "all_labels.npy"))
    all_embedding = np.load(os.path.join(features_dir, "umap_embedding.npy"))
    ndvii_scores = np.load(os.path.join(features_dir, "ndvii_scores.npy"))

    # Fit scaler and UMAP on existing data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    reducer.fit(features_scaled)

    # Control centroid for NDVII
    control_mask = all_labels == 0
    control_centroid = all_embedding[control_mask].mean(axis=0)
    max_dist = np.linalg.norm(all_embedding - control_centroid, axis=1).max()
    min_dist_val = np.linalg.norm(all_embedding - control_centroid, axis=1).min()

    return (
        model, cfg, device, scaler, reducer,
        all_embedding, all_labels, ndvii_scores,
        control_centroid, min_dist_val, max_dist
    )


def get_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg["preprocessing"]["normalize_mean"],
            std=cfg["preprocessing"]["normalize_std"]
        ),
    ])


def predict_single(image, model, cfg, device, scaler, reducer,
                   control_centroid, min_dist_val, max_dist):
    transform = get_transform(cfg)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, features = model(img_tensor, return_features=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        features_np = features.cpu().numpy()

    # Project into embedding space
    features_scaled = scaler.transform(features_np)
    new_embedding = reducer.transform(features_scaled)

    # Compute NDVII
    dist = np.linalg.norm(new_embedding[0] - control_centroid)
    ndvii = (dist - min_dist_val) / (max_dist - min_dist_val)
    ndvii = float(np.clip(ndvii, 0.0, 1.0))

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    class_names = ["Control Group", "DM Group"]

    return {
        "ndvii": ndvii,
        "stability": classify_ndvii(ndvii),
        "pred_class": class_names[pred_class],
        "confidence": confidence,
        "embedding": new_embedding[0],
        "features": features_np[0],
    }


def plot_embedding_with_new(all_embedding, all_labels, new_point):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1320")
    ax.set_facecolor("#0d1320")

    colors = {0: "#00e5ff", 1: "#ff4b6e"}
    names = {0: "Control Group", 1: "DM Group"}

    for cls in [0, 1]:
        mask = all_labels == cls
        ax.scatter(
            all_embedding[mask, 0], all_embedding[mask, 1],
            c=colors[cls], label=names[cls],
            s=12, alpha=0.5, edgecolors="none"
        )

    # New point
    ax.scatter(
        new_point[0], new_point[1],
        c="#ffffff", s=180, marker="*",
        zorder=5, label="Your Image", edgecolors="#ffb300",
        linewidths=1.5
    )

    ax.set_title("PSE Embedding Space — Your Image Location",
                 color="white", fontsize=11, pad=10)
    ax.set_xlabel("UMAP-1", color="#5a7090", fontsize=9)
    ax.set_ylabel("UMAP-2", color="#5a7090", fontsize=9)
    ax.tick_params(colors="#5a7090", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d47")

    legend = ax.legend(
        facecolor="#131b2e", edgecolor="#1e2d47",
        labelcolor="white", fontsize=8
    )
    plt.tight_layout()
    return fig


def ndvii_color_class(stability):
    mapping = {
        "Thermally Stable": "ndvii-stable",
        "Mild Instability": "ndvii-mild",
        "Moderate Instability": "ndvii-moderate",
        "High Instability": "ndvii-high",
    }
    return mapping.get(stability, "")


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌡️ ThermalRiskAI")
    st.markdown("---")
    st.markdown("**Research Platform**")
    st.markdown("Thermal stability analysis of diabetic foot thermography.")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("- EfficientNet-B0 CNN")
    st.markdown("- 128-dim feature extraction")
    st.markdown("- UMAP PSE embedding")
    st.markdown("- NDVII score computation")
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.markdown("- Val Accuracy: **92.4%**")
    st.markdown("- NDVII Accuracy: **98%**")
    st.markdown("- Separation Gap: **0.69**")
    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ Research use only. Not a diagnostic tool.</div>',
        unsafe_allow_html=True
    )


# ─── Main Layout ───────────────────────────────────────────────────────────────
st.markdown("# ThermalRiskAI")
st.markdown("##### Perfusion Stability Analysis Platform — Research Edition")
st.markdown("---")

# Load everything
with st.spinner("Loading model and embeddings..."):
    (model, cfg, device, scaler, reducer,
     all_embedding, all_labels, ndvii_scores,
     control_centroid, min_dist_val, max_dist) = load_model_and_data()

st.success("Model loaded. Ready for analysis.")

# ─── Upload Section ─────────────────────────────────────────────────────────────
st.markdown("### Upload Thermal Image")
uploaded_file = st.file_uploader(
    "Upload a thermal foot image (.png or .jpg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Uploaded Image**")
        st.image(image, use_column_width=True)

    with st.spinner("Analyzing..."):
        result = predict_single(
            image, model, cfg, device, scaler, reducer,
            control_centroid, min_dist_val, max_dist
        )

    with col2:
        st.markdown("**Analysis Results**")

        m1, m2, m3 = st.columns(3)

        ndvii_val = result["ndvii"]
        stability = result["stability"]
        css_class = ndvii_color_class(stability)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {css_class}">{ndvii_val:.3f}</div>
                <div class="metric-label">NDVII Score</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {css_class}" style="font-size:1.1rem;padding-top:8px;">{stability}</div>
                <div class="metric-label">Stability Class</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            conf_pct = f"{result['confidence']*100:.1f}%"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{conf_pct}</div>
                <div class="metric-label">Model Confidence</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        pred = result["pred_class"]
        st.markdown(f"**Predicted Group:** `{pred}`")
        st.markdown(f"**Embedding coordinates:** `UMAP-1: {result['embedding'][0]:.3f}` | `UMAP-2: {result['embedding'][1]:.3f}`")

    # Embedding plot
    st.markdown("### PSE Embedding — Image Location")
    fig = plot_embedding_with_new(all_embedding, all_labels, result["embedding"])
    st.pyplot(fig)
    plt.close()

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Research Platform — Non-Diagnostic:</strong>
    All outputs including NDVII scores and stability classifications are
    computational research indicators only. They are not validated clinical
    biomarkers and must not be used for medical decision-making.
    </div>
    """, unsafe_allow_html=True)

else:
    # Show precomputed embedding when no image uploaded
    st.markdown("### Dataset PSE Embedding")
    st.caption("Upload an image above to see where it falls in the embedding space.")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1320")
    ax.set_facecolor("#0d1320")

    colors = {0: "#00e5ff", 1: "#ff4b6e"}
    names = {0: "Control Group", 1: "DM Group"}

    for cls in [0, 1]:
        mask = all_labels == cls
        ax.scatter(
            all_embedding[mask, 0], all_embedding[mask, 1],
            c=colors[cls], label=names[cls],
            s=14, alpha=0.6, edgecolors="none"
        )

    ax.set_title("Perfusion Stability Embedding (PSE) — Full Dataset",
                 color="white", fontsize=12, pad=12)
    ax.set_xlabel("UMAP-1", color="#5a7090")
    ax.set_ylabel("UMAP-2", color="#5a7090")
    ax.tick_params(colors="#5a7090")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d47")

    ax.legend(facecolor="#131b2e", edgecolor="#1e2d47", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()