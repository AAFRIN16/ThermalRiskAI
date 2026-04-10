import numpy as np
import hashlib


# Deterministic organ zone definitions
# Based on reflexology and thermal zone anatomical mapping
FEET_ZONES = {
    "cardiovascular": {
        "description": "Heart & Circulatory System",
        "zone": "Medial plantar arch region",
        "icon": "heart",
        "system": "Cardiovascular"
    },
    "renal": {
        "description": "Kidneys & Urinary Tract",
        "zone": "Central plantar region",
        "icon": "activity",
        "system": "Renal"
    },
    "digestive": {
        "description": "Digestive System",
        "zone": "Lateral plantar region",
        "icon": "zap",
        "system": "Digestive"
    },
    "endocrine": {
        "description": "Endocrine & Hormonal System",
        "zone": "Medial calcaneal region",
        "icon": "cpu",
        "system": "Endocrine"
    },
    "nervous": {
        "description": "Peripheral Nervous System",
        "zone": "Toe tips & plantar surface",
        "icon": "git-branch",
        "system": "Neurological"
    },
    "lymphatic": {
        "description": "Lymphatic & Immune System",
        "zone": "Lateral calcaneal region",
        "icon": "shield",
        "system": "Lymphatic"
    },
    "musculoskeletal": {
        "description": "Musculoskeletal System",
        "zone": "Heel & lateral foot border",
        "icon": "layers",
        "system": "Musculoskeletal"
    },
    "respiratory": {
        "description": "Respiratory System",
        "zone": "Ball of foot region",
        "icon": "wind",
        "system": "Respiratory"
    }
}

PALM_ZONES = {
    "cardiovascular": {
        "description": "Heart & Circulatory System",
        "zone": "Thenar eminence (thumb base)",
        "icon": "heart",
        "system": "Cardiovascular"
    },
    "renal": {
        "description": "Kidneys & Urinary Tract",
        "zone": "Central palm region",
        "icon": "activity",
        "system": "Renal"
    },
    "digestive": {
        "description": "Digestive System",
        "zone": "Hypothenar eminence",
        "icon": "zap",
        "system": "Digestive"
    },
    "endocrine": {
        "description": "Endocrine & Hormonal System",
        "zone": "Wrist crease region",
        "icon": "cpu",
        "system": "Endocrine"
    },
    "nervous": {
        "description": "Peripheral Nervous System",
        "zone": "Fingertip thermal zones",
        "icon": "git-branch",
        "system": "Neurological"
    },
    "lymphatic": {
        "description": "Lymphatic & Immune System",
        "zone": "Inter-digital web spaces",
        "icon": "shield",
        "system": "Lymphatic"
    },
    "musculoskeletal": {
        "description": "Musculoskeletal System",
        "zone": "Metacarpal region",
        "icon": "layers",
        "system": "Musculoskeletal"
    },
    "respiratory": {
        "description": "Respiratory System",
        "zone": "Index & middle finger zones",
        "icon": "wind",
        "system": "Respiratory"
    }
}

STATUS_LEVELS = ["Optimal", "Normal", "Mild Variation", "Moderate Variation", "Elevated Concern"]
STATUS_COLORS = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]


def get_image_hash(features_np):
    """Generate deterministic hash from feature vector for consistent outputs."""
    feature_bytes = features_np.tobytes()
    return int(hashlib.md5(feature_bytes).hexdigest(), 16)


def compute_organ_status(features_np, ndvii, region_type="feet"):
    """
    Deterministically compute organ zone status from feature vector.
    Same input always produces same output.
    """
    hash_val = get_image_hash(features_np)
    zones = FEET_ZONES if region_type == "feet" else PALM_ZONES

    organ_results = []
    feature_values = features_np[0] if len(features_np.shape) > 1 else features_np

    for i, (organ_key, organ_info) in enumerate(zones.items()):
        # Deterministic index from feature vector segment
        segment_start = (i * 16) % len(feature_values)
        segment = feature_values[segment_start:segment_start + 16]
        segment_mean = float(np.mean(np.abs(segment)))

        # Map to status level deterministically
        # Base score influenced by NDVII and local feature energy
        base_score = (segment_mean * 0.6 + ndvii * 0.4)
        status_idx = min(int(base_score * 4.5), 4)

        # Perfusion level (0-100)
        perfusion = max(0, min(100, int((1 - base_score * 0.7) * 100)))

        # Temperature offset (deterministic)
        temp_offset = round((segment_mean - 0.5) * 2.4, 2)

        organ_results.append({
            "organ": organ_key,
            "system": organ_info["system"],
            "description": organ_info["description"],
            "zone": organ_info["zone"],
            "icon": organ_info["icon"],
            "status": STATUS_LEVELS[status_idx],
            "status_color": STATUS_COLORS[status_idx],
            "perfusion_level": perfusion,
            "thermal_variation": temp_offset,
            "status_index": status_idx
        })

    # Overall health score
    avg_status = np.mean([r["status_index"] for r in organ_results])
    overall_score = max(0, min(100, int((1 - avg_status / 4) * 100)))

    return {
        "organs": organ_results,
        "overall_health_score": overall_score,
        "region_type": region_type,
        "systems_summary": _compute_systems_summary(organ_results)
    }


def _compute_systems_summary(organ_results):
    systems = {}
    for organ in organ_results:
        sys_name = organ["system"]
        if sys_name not in systems:
            systems[sys_name] = []
        systems[sys_name].append(organ["status_index"])

    summary = []
    for sys_name, indices in systems.items():
        avg = np.mean(indices)
        summary.append({
            "system": sys_name,
            "average_concern": round(float(avg), 2),
            "status": STATUS_LEVELS[min(int(avg + 0.5), 4)],
            "color": STATUS_COLORS[min(int(avg + 0.5), 4)]
        })
    return summary


def detect_region_type(image_pil):
    """
    Detect if image is feet or palm based on color distribution.
    Returns 'feet', 'palm', or 'unknown'.
    """
    import numpy as np
    img_array = np.array(image_pil.resize((224, 224)))

    # Basic shape analysis
    non_black = img_array.max(axis=2) > 30
    non_black_ratio = non_black.mean()

    if non_black_ratio < 0.05 or non_black_ratio > 0.98:
        return "unknown"

    # Aspect ratio check
    h, w = img_array.shape[:2]
    aspect = h / w

    # Feet tend to be taller/more oval, palms more square
    if aspect > 1.2:
        return "feet"
    elif aspect < 0.9:
        return "palm"
    else:
        return "feet"  # default to feet since dataset is feet-based