import os
import gdown

FILES = {
    "outputs/models/best_model.pth": "1TrCsfCckpf6hYD-RBoNhSbTF_mSb3gRs",
    "outputs/features/all_features.npy": "12RuqTl2pSNpdrwh-fH16ewp0FOoQ1Yym",
    "outputs/features/all_labels.npy": "12yRBd1h-swzzoGUAYKz-_n1E4IufcRqC",
    "outputs/features/umap_embedding.npy": "19spQWB3jCvEx2GMjxt8_C_ZA1xixouYz",
}

def download_all():
    for path, file_id in FILES.items():
        if os.path.exists(path):
            print(f"Already exists: {path}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {path}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
        print(f"Done: {path}")

if __name__ == "__main__":
    download_all()