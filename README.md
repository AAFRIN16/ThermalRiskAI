---
title: ThermalRiskAI
emoji: 🌡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# ThermalRiskAI

**AI-Based Spatial–Temporal Thermal Stability Analysis Platform**

A research-oriented system that analyzes infrared thermal images of diabetic feet to study perfusion-related heat patterns through computational modeling and spatial-temporal stability analysis.

> ⚠️ **Non-Diagnostic:** All outputs are computational research indicators only and are not intended for medical diagnosis.

---

## Project Structure
```
ThermalRiskAI/                  # Backend (FastAPI + PyTorch)
├── data/ThermoDataBase/        # Dataset (train/ and val/)
├── src/
│   ├── dataset.py              # DataLoader and transforms
│   ├── model.py                # EfficientNet-B0 model
│   ├── train.py                # Training loop with checkpointing
│   ├── features.py             # Feature extraction
│   ├── embedding.py            # UMAP PSE embedding
│   └── ndvii.py                # NDVII score computation
├── app/
│   ├── api.py                  # FastAPI backend
│   └── dashboard.py            # Streamlit prototype
├── outputs/
│   ├── models/best_model.pth   # Trained model weights
│   ├── features/               # Extracted features + embeddings
│   └── logs/                   # Training history
└── configs/config.yaml         # All configuration

thermalriskai-frontend/         # Frontend (React + TypeScript + Tailwind)
├── src/
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── Upload.tsx
│   │   └── About.tsx
│   └── components/
│       └── Navbar.tsx
└── dist/                       # Production build
```

---

## Model Performance

| Metric | Value |
|---|---|
| Validation Accuracy | 92.4% |
| NDVII Accuracy | 98% |
| Separation Gap | 0.69 |
| Backbone | EfficientNet-B0 |
| Training Epochs | 15 (early stopping) |

---

## Pipeline
```
Infrared Image → Preprocessing → CNN Feature Extraction (128-dim)
→ UMAP PSE Embedding → NDVII Score → Dashboard Visualization
```

---

## Setup & Running

### Backend
```bash
cd ThermalRiskAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start API
uvicorn app.api:app --reload --port 8000
```

### Frontend
```bash
cd thermalriskai-frontend
npm install
npm run dev
```

Open `http://localhost:5173`

### Train from scratch
```bash
python -m src.train
python -m src.features
python -m src.embedding
python -m src.ndvii
```

---

## Dataset

**Thermography Images of Diabetic Foot**
- Source: [Kaggle](https://www.kaggle.com/datasets/vuppalaadithyasairam/thermography-images-of-diabetic-foot)
- 1,866 PNG images
- Classes: Control Group / DM Group
- Pre-split into train (1,444) and val (422)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch + EfficientNet-B0 |
| Embedding | UMAP |
| API | FastAPI |
| Frontend | React + TypeScript + Tailwind CSS |
| Charts | Recharts |
| Visualization | Matplotlib |

---

## NDVII Score Reference

| Range | Label |
|---|---|
| 0.00 – 0.30 | Thermally Stable |
| 0.30 – 0.55 | Mild Instability |
| 0.55 – 0.75 | Moderate Instability |
| 0.75 – 1.00 | High Instability |

---