import torch
import torch.nn as nn
from torchvision import models
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ThermalRiskModel(nn.Module):
    def __init__(self, num_classes=2, feature_dim=128, dropout=0.3, pretrained=True):
        super(ThermalRiskModel, self).__init__()

        # Load EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Get the in_features of the original classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with custom head
        # This gives us: backbone features → feature_dim → class logits
        self.backbone.classifier = nn.Identity()

        self.feature_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x, return_features=False):
        # Extract backbone features
        x = self.backbone(x)

        # Pass through feature head → 128-dim spatial feature vector
        features = self.feature_head(x)

        # Classification logits
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


def build_model(cfg):
    model = ThermalRiskModel(
        num_classes=cfg["model"]["num_classes"],
        feature_dim=cfg["model"]["feature_dim"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
    )
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")


if __name__ == "__main__":
    cfg = load_config()
    model = build_model(cfg)
    count_parameters(model)

    # Test forward pass
    dummy = torch.randn(4, 3, 224, 224)
    logits, features = model(dummy, return_features=True)
    print(f"Logits shape:   {logits.shape}")
    print(f"Features shape: {features.shape}")
    print("Model build successful.")