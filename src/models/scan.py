from datetime import datetime
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field


class ThermalScanModel(BaseModel):
    scanId: str = Field(..., description="Unique scan identifier")
    userId: str = Field(..., description="Owner user ID")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Scan timestamp")
    uploadedImage: str = Field(default="", description="Base64 or image filename/path reference")
    prediction: Dict[str, Any] = Field(default_factory=dict, description="Model prediction output (class, confidence)")
    NDVII: Union[float, Dict[str, Any]] = Field(..., description="NDVII value or stability metrics breakdown")
    OrganMapping: Union[list, Dict[str, Any]] = Field(default_factory=dict, description="Organ mapping thermal features")
    GradCAM: str = Field(default="", description="GradCAM visualization base64 image")
    WellnessScore: float = Field(..., description="Calculated thermal wellness score (0-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "scanId": "scan_20260808_001",
                "userId": "usr_123456",
                "timestamp": "2026-08-08T14:35:00Z",
                "uploadedImage": "data:image/jpeg;base64,...",
                "prediction": {"predicted_class": "Control Group", "confidence": 94.2},
                "NDVII": {"ndvii": 0.12, "stability_score": 88.0, "stability_label": "Thermally Stable"},
                "OrganMapping": {"pancreas": {"status": "Stable", "temp": 33.2}},
                "GradCAM": "data:image/png;base64,...",
                "WellnessScore": 88.0
            }
        }
