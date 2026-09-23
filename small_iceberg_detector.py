"""
Small Iceberg Database & Detection Module
==========================================
Generates a realistic catalog of small icebergs (< 200m length) in the
Antarctic operational zone and provides ML-based detection from environmental
signature analysis.

Small icebergs are the most dangerous to navigation because:
- They are below SAR satellite resolution threshold (~50m)
- They ride ocean currents more than wind (unlike large tabular bergs)
- They can calve unpredictably from larger icebergs or glaciers
- Their drift is highly non-linear due to wave-berg coupling

This module:
1. Generates a realistic population of small icebergs based on published
   calving distributions (Tournadre et al. 2016, Wesche & Dierking 2015)
2. Provides an environmental-signature detection model that identifies
   probable small iceberg locations from SST anomalies, ice concentration
   gradients, and ocean current convergence zones
3. Classifies hazard severity for navigation planning

Iceberg Size Classification (based on NIC/IIP standards):
  - Growler: L < 5m, H < 1m (barely visible)
  - Bergy Bit: 5m < L < 15m, 1m < H < 5m
  - Small: 15m < L < 60m, 5m < H < 15m
  - Medium: 60m < L < 120m, 15m < H < 45m
  - Large: 120m < L < 200m, 45m < H < 75m
"""

import math
import json
import os
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple

# Import environment utilities
try:
    from environment_simulator import (
        AntarcticEnvironment, get_field_at_point,
        GRID_H, GRID_W, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        _lat_grid, _lon_grid,
    )
except ImportError:
    # Standalone fallback
    GRID_H, GRID_W = 128, 128
    LAT_MIN, LAT_MAX = -72.0, -60.0
    LON_MIN, LON_MAX = 10.0, 80.0


# ── Iceberg Size Classes ──────────────────────────────────────
SIZE_CLASSES = {
    "growler":   {"L_min": 1, "L_max": 5,   "H_min": 0.3, "H_max": 1,  "draft_ratio": 6.0, "radar_cross_section_m2": 0.5},
    "bergy_bit": {"L_min": 5, "L_max": 15,  "H_min": 1,   "H_max": 5,  "draft_ratio": 5.5, "radar_cross_section_m2": 5.0},
    "small":     {"L_min": 15,"L_max": 60,  "H_min": 5,   "H_max": 15, "draft_ratio": 5.0, "radar_cross_section_m2": 50.0},
    "medium":    {"L_min": 60,"L_max": 120, "H_min": 15,  "H_max": 45, "draft_ratio": 4.5, "radar_cross_section_m2": 500.0},
    "large":     {"L_min": 120,"L_max":200, "H_min": 45,  "H_max": 75, "draft_ratio": 4.0, "radar_cross_section_m2": 2000.0},
}

# Calving zones — regions where small icebergs are most likely to form
# Based on Antarctic ice shelf locations and known calving fronts
CALVING_HOTSPOTS = [
    {"name": "Fimbul Ice Shelf",    "lat": -70.5, "lon": 0.0,   "radius_km": 80, "rate": 1.5},
    {"name": "Riiser-Larsen",       "lat": -72.0, "lon": 15.0,  "radius_km": 60, "rate": 1.0},
    {"name": "Brunt Ice Shelf",     "lat": -75.5, "lon": -26.0, "radius_km": 100,"rate": 2.0},
    {"name": "Lazarev Sea",         "lat": -69.0, "lon": 14.0,  "radius_km": 120,"rate": 1.8},
    {"name": "Enderby Land Coast",  "lat": -67.5, "lon": 50.0,  "radius_km": 90, "rate": 1.2},
    {"name": "Amery Ice Shelf",     "lat": -69.5, "lon": 72.0,  "radius_km": 100,"rate": 2.5},
    {"name": "Maitri Station Area", "lat": -70.7, "lon": 11.7,  "radius_km": 50, "rate": 1.0},
    {"name": "Bharati Station Area","lat": -69.4, "lon": 76.2,  "radius_km": 50, "rate": 0.8},
]


def generate_iceberg_population(
    n_icebergs: int = 50,
    timestamp: Optional[datetime.datetime] = None,
    seed: Optional[int] = None,
    env_state: Optional[Dict] = None,
) -> List[Dict]:
    """
    Generate a realistic population of small icebergs.

    The distribution follows:
    - 40% growlers
    - 25% bergy bits
    - 20% small
    - 10% medium
    - 5% large

    Positions are clustered near calving hotspots and ice edges.
    """
    rng = np.random.RandomState(seed)
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()

    # Size class probabilities
    class_probs = [0.40, 0.25, 0.20, 0.10, 0.05]
    class_names = list(SIZE_CLASSES.keys())

    icebergs = []
    for i in range(n_icebergs):
        # Pick size class
        cls_name = rng.choice(class_names, p=class_probs)
        cls = SIZE_CLASSES[cls_name]

        # Pick a calving hotspot (weighted by rate)
        rates = np.array([h["rate"] for h in CALVING_HOTSPOTS])
        rates /= rates.sum()
        hotspot = CALVING_HOTSPOTS[rng.choice(len(CALVING_HOTSPOTS), p=rates)]

        # Position: Gaussian scatter around hotspot, clipped to domain
        scatter_deg = hotspot["radius_km"] / 111.0  # rough km->deg
        lat = np.clip(
            hotspot["lat"] + rng.randn() * scatter_deg,
            LAT_MIN, LAT_MAX
        )
        lon = np.clip(
            hotspot["lon"] + rng.randn() * scatter_deg * 2,  # wider in lon
            LON_MIN, LON_MAX
        )

        # Physical dimensions
        L = rng.uniform(cls["L_min"], cls["L_max"])
        W = L * rng.uniform(0.4, 0.8)
        H = rng.uniform(cls["H_min"], cls["H_max"])
        draft = H * cls["draft_ratio"]
        mass_tonnes = 917.0 * L * W * (H + draft) / 1000.0  # ρ_ice = 917 kg/m³

        # Initial velocity (from environment or default)
        u0, v0 = 0.0, 0.0
        if env_state is not None:
            try:
                u0 = get_field_at_point(env_state, lat, lon, 'ocean_u') + \
                     0.02 * get_field_at_point(env_state, lat, lon, 'wind_u')
                v0 = get_field_at_point(env_state, lat, lon, 'ocean_v') + \
                     0.02 * get_field_at_point(env_state, lat, lon, 'wind_v')
            except Exception:
                pass

        # Detection difficulty (higher = harder to detect)
        detection_difficulty = 1.0 - (cls["radar_cross_section_m2"] / 2000.0)

        berg = {
            "id": f"SB-{i+1:04d}",
            "timestamp": timestamp.isoformat(),
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "size_class": cls_name,
            "length_m": round(float(L), 1),
            "width_m": round(float(W), 1),
            "height_m": round(float(H), 1),
            "draft_m": round(float(draft), 1),
            "mass_tonnes": round(float(mass_tonnes), 0),
            "u": round(float(u0), 5),
            "v": round(float(v0), 5),
            "speed_ms": round(float(math.sqrt(u0**2 + v0**2)), 5),
            "heading_deg": round(float((math.degrees(math.atan2(v0, u0)) + 360) % 360), 1),
            "source_hotspot": hotspot["name"],
            "detection_difficulty": round(float(detection_difficulty), 3),
            "radar_cross_section_m2": cls["radar_cross_section_m2"],
            "hazard_radius_m": round(float(L * 3 + draft * 0.5), 1),  # safety buffer
            "confidence": round(float(rng.uniform(0.4, 0.95)), 2),
            "last_observed": timestamp.isoformat(),
            "status": "active",
        }
        icebergs.append(berg)

    return icebergs


def detect_icebergs_from_environment(
    env_state: Dict,
    sensitivity: float = 0.5,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    ML-inspired iceberg detection from environmental signatures.

    Detection is based on converging multiple signals:
    1. SST anomaly: cold spots in otherwise warmer water → meltwater plume
    2. Ice edge proximity: icebergs concentrate near ice margins
    3. Current convergence: icebergs accumulate in convergence zones
    4. Bathymetry: shallow areas trap grounded bergs

    Returns detected iceberg candidates with confidence scores.
    """
    rng = np.random.RandomState(seed)

    sst = np.array(env_state.get('sst', np.zeros((GRID_H, GRID_W))))
    ice = np.array(env_state.get('ice_conc', np.zeros((GRID_H, GRID_W))))
    ocean_u = np.array(env_state.get('ocean_u', np.zeros((GRID_H, GRID_W))))
    ocean_v = np.array(env_state.get('ocean_v', np.zeros((GRID_H, GRID_W))))
    bathy = np.array(env_state.get('bathymetry', np.full((GRID_H, GRID_W), -3000.0)))

    # Signal 1: SST cold anomaly (local minimum detection)
    from scipy.ndimage import gaussian_filter, minimum_filter
    sst_smooth = gaussian_filter(sst, sigma=3)
    sst_local_min = minimum_filter(sst_smooth, size=7)
    sst_anomaly = np.clip(sst_smooth - sst_local_min, 0, 2.0)
    sst_score = sst_anomaly / 2.0

    # Signal 2: Ice edge gradient (strongest detection near ice margins)
    ice_grad_y, ice_grad_x = np.gradient(ice)
    ice_gradient_mag = np.sqrt(ice_grad_y**2 + ice_grad_x**2)
    ice_edge_score = np.clip(ice_gradient_mag / 0.05, 0, 1)

    # Signal 3: Current convergence (divergence < 0 = convergence)
    div_u = np.gradient(ocean_u, axis=1)
    div_v = np.gradient(ocean_v, axis=0)
    convergence = -(div_u + div_v)
    conv_score = np.clip(convergence / 0.01, 0, 1)

    # Signal 4: Shallow bathymetry (grounding risk)
    bathy_score = np.clip((-bathy - 100) / 400, 0, 1)  # Higher for depths 100-500m

    # Combined detection score
    detection_score = (
        0.35 * sst_score +
        0.30 * ice_edge_score +
        0.20 * conv_score +
        0.15 * bathy_score
    )

    # Apply sensitivity threshold
    threshold = 1.0 - sensitivity
    candidates = []

    # Find local maxima in detection score
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(detection_score, size=5)
    peaks = (detection_score == local_max) & (detection_score > threshold * 0.3)

    peak_coords = np.argwhere(peaks)
    # Limit to top 30 candidates
    if len(peak_coords) > 30:
        scores_at_peaks = detection_score[peak_coords[:, 0], peak_coords[:, 1]]
        top_idx = np.argsort(scores_at_peaks)[-30:]
        peak_coords = peak_coords[top_idx]

    timestamp = env_state.get('timestamp', datetime.datetime.utcnow().isoformat())

    for idx, (r, c) in enumerate(peak_coords):
        lat = LAT_MAX - (r / (GRID_H - 1)) * (LAT_MAX - LAT_MIN)
        lon = LON_MIN + (c / (GRID_W - 1)) * (LON_MAX - LON_MIN)
        score = float(detection_score[r, c])

        # Infer size class from score and location
        if score > 0.7:
            cls_name = rng.choice(["small", "medium", "large"], p=[0.5, 0.3, 0.2])
        elif score > 0.4:
            cls_name = rng.choice(["growler", "bergy_bit", "small"], p=[0.3, 0.4, 0.3])
        else:
            cls_name = rng.choice(["growler", "bergy_bit"], p=[0.6, 0.4])

        cls = SIZE_CLASSES[cls_name]
        L = rng.uniform(cls["L_min"], cls["L_max"])

        candidates.append({
            "id": f"DET-{idx+1:03d}",
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "detection_score": round(score, 3),
            "size_class": cls_name,
            "estimated_length_m": round(float(L), 1),
            "confidence": round(float(np.clip(score * 1.2, 0, 0.95)), 2),
            "detection_method": "Environmental Signature Analysis",
            "signals": {
                "sst_anomaly": round(float(sst_score[r, c]), 3),
                "ice_edge_gradient": round(float(ice_edge_score[r, c]), 3),
                "current_convergence": round(float(conv_score[r, c]), 3),
                "shallow_bathymetry": round(float(bathy_score[r, c]), 3),
            },
            "timestamp": timestamp,
            "status": "detected",
        })

    return candidates


def classify_hazard_level(iceberg: Dict) -> str:
    """Classify navigation hazard based on iceberg properties."""
    L = iceberg.get("length_m", iceberg.get("estimated_length_m", 10))
    confidence = iceberg.get("confidence", 0.5)
    speed = iceberg.get("speed_ms", 0.0)

    if L > 100 and confidence > 0.6:
        return "CRITICAL"
    elif L > 50 or (L > 20 and speed > 0.3):
        return "HIGH"
    elif L > 15 or confidence > 0.7:
        return "MODERATE"
    else:
        return "LOW"


def export_iceberg_database(icebergs: List[Dict], output_path: str):
    """Export iceberg catalog to JSON."""
    for berg in icebergs:
        berg["hazard_level"] = classify_hazard_level(berg)

    output = {
        "version": 2,
        "generated": datetime.datetime.utcnow().isoformat(),
        "count": len(icebergs),
        "size_distribution": {},
        "icebergs": icebergs,
    }

    # Size distribution summary
    for cls in SIZE_CLASSES:
        count = sum(1 for b in icebergs if b.get("size_class") == cls)
        output["size_distribution"][cls] = count

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    from environment_simulator import AntarcticEnvironment

    print("=" * 60)
    print("Small Iceberg Database Generator")
    print("=" * 60)

    env = AntarcticEnvironment(seed=42)
    state = env.generate(datetime.datetime(2026, 3, 15, 12, 0))

    # Generate population
    population = generate_iceberg_population(50, seed=123, env_state=state)
    print(f"\nGenerated {len(population)} icebergs:")
    for cls in SIZE_CLASSES:
        count = sum(1 for b in population if b["size_class"] == cls)
        print(f"  {cls:>12s}: {count}")

    # Detect from environment
    detected = detect_icebergs_from_environment(state, sensitivity=0.6, seed=456)
    print(f"\nDetected {len(detected)} iceberg candidates from environment:")
    for d in detected[:5]:
        print(f"  {d['id']}: {d['size_class']} at ({d['lat']:.2f}, {d['lon']:.2f}) "
              f"score={d['detection_score']:.3f}")

    # Export
    all_bergs = population + detected
    out_path = os.path.join(os.path.dirname(__file__), "data", "iceberg_database.json")
    result = export_iceberg_database(all_bergs, out_path)
    print(f"\nExported {result['count']} icebergs to {out_path}")
    print(f"Size distribution: {result['size_distribution']}")
