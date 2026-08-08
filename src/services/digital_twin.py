from typing import List, Dict, Any, Optional
from src.models.scan import ThermalScanModel


class DigitalTwinService:
    """
    Computes Digital Twin state from a user's thermal scan history.
    Designed to be clean, modular, and easily extendable for future AI features.
    """

    @staticmethod
    def _extract_ndvii(scan: ThermalScanModel) -> float:
        if isinstance(scan.NDVII, dict):
            return float(scan.NDVII.get("ndvii", 0.0))
        elif isinstance(scan.NDVII, (float, int)):
            return float(scan.NDVII)
        return 0.0

    @classmethod
    def compute_digital_twin(cls, user_id: str, scans: List[ThermalScanModel]) -> Dict[str, Any]:
        scans_count = len(scans)

        if scans_count == 0:
            return {
                "digital_twin_summary": {
                    "user_id": user_id,
                    "scans_processed": 0,
                    "avg_wellness_score": 0.0,
                    "avg_ndvii": 0.0,
                    "baseline_status": "Not Established",
                    "recovery_trend": "0.0%"
                },
                "trend": "Awaiting initial thermal scan",
                "latest_scan": None,
                "historical_metrics": []
            }

        # Computations
        wellness_scores = [scan.WellnessScore for scan in scans]
        ndvii_values = [cls._extract_ndvii(scan) for scan in scans]

        avg_wellness = round(sum(wellness_scores) / scans_count, 1)
        avg_ndvii = round(sum(ndvii_values) / scans_count, 4)

        # Sort chronologically (oldest to newest) for trend analysis
        chronological_scans = sorted(scans, key=lambda s: s.timestamp)
        latest_scan = chronological_scans[-1]

        # Baseline Status
        if scans_count >= 3:
            baseline_status = "Established"
        elif scans_count >= 1:
            baseline_status = "Calibrating"
        else:
            baseline_status = "Not Established"

        # Trend & Recovery Calculation
        if scans_count < 2:
            trend = "Baseline calibration active"
            recovery_trend = "0.0%"
        else:
            first_wellness = chronological_scans[0].WellnessScore
            last_wellness = chronological_scans[-1].WellnessScore
            diff = last_wellness - first_wellness

            pct_change = (diff / first_wellness * 100.0) if first_wellness > 0 else 0.0
            recovery_trend = f"{'+' if pct_change >= 0 else ''}{round(pct_change, 1)}%"

            if diff > 2.0:
                trend = "Improving thermal stability"
            elif diff < -2.0:
                trend = "Declining thermal stability"
            else:
                trend = "Stable thermal profile"

        # Historical Metrics for frontend timeline & charts
        historical_metrics = []
        for s in chronological_scans:
            historical_metrics.append({
                "scan_id": s.scanId,
                "timestamp": s.timestamp,
                "wellness_score": s.WellnessScore,
                "ndvii": cls._extract_ndvii(s),
                "predicted_class": s.prediction.get("predicted_class") if isinstance(s.prediction, dict) else "Unknown",
                "confidence": s.prediction.get("confidence") if isinstance(s.prediction, dict) else 0.0
            })

        return {
            "digital_twin_summary": {
                "user_id": user_id,
                "scans_processed": scans_count,
                "avg_wellness_score": avg_wellness,
                "avg_ndvii": avg_ndvii,
                "baseline_status": baseline_status,
                "recovery_trend": recovery_trend,
            },
            "trend": trend,
            "latest_scan": latest_scan.model_dump() if latest_scan else None,
            "historical_metrics": historical_metrics
        }
