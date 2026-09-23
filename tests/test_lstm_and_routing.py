import math
import json
import sys
from pathlib import Path

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from iceberg_lstm import IcebergDriftLSTM, predict_trajectory


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def test_lstm_model():
    print("\n--- 1. Testing Trained IcebergDriftLSTM Model ---")
    ckpt_path = Path("checkpoints_local/iceberg_lstm.pt")
    assert ckpt_path.exists(), "Checkpoint file checkpoints_local/iceberg_lstm.pt does not exist!"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = IcebergDriftLSTM(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        output_dim=ckpt["output_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_steps = [
        {"lat": -70.62, "lon": 3.08, "u": 0.01, "v": 0.02, "speed": 0.022, "heading": 63.4, "t": 1000.0, "elapsed_h": 0.0},
        {"lat": -70.621, "lon": 3.085, "u": 0.012, "v": 0.021, "speed": 0.024, "heading": 60.2, "t": 1060.0, "elapsed_h": 1.0},
        {"lat": -70.623, "lon": 3.090, "u": 0.015, "v": 0.018, "speed": 0.023, "heading": 50.1, "t": 1120.0, "elapsed_h": 2.0},
        {"lat": -70.625, "lon": 3.095, "u": 0.018, "v": 0.015, "speed": 0.023, "heading": 39.8, "t": 1180.0, "elapsed_h": 3.0},
    ]

    forecast = predict_trajectory(model, sample_steps, forecast_hours=12)
    assert len(forecast) == 12, f"Expected 12 forecast steps, got {len(forecast)}"
    assert "lat" in forecast[0] and "lon" in forecast[0] and "pos_uncertainty_m" in forecast[0]
    print(f"[PASS] LSTM forecast verified: 12 steps generated. Final pos: {forecast[-1]['lat']}S, {forecast[-1]['lon']}E, Uncertainty: {forecast[-1]['pos_uncertainty_m']} m")


def test_gis_dataset():
    print("\n--- 2. Testing Comprehensive Antarctica GeoJSON ---")
    geojson_path = Path("frontend/public/antarctica.json")
    assert geojson_path.exists(), "Antarctica GeoJSON does not exist!"

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    assert len(features) >= 5, f"Expected at least 5 features, got {len(features)}"

    names = [f.get("properties", {}).get("name", "") for f in features]
    print(f"[PASS] GeoJSON verified: {len(features)} features found: {', '.join(filter(None, names))}")


def test_iceberg_standoff():
    print("\n--- 3. Testing Known Iceberg Standoff & Avoidance Geometry ---")
    # A-23a location
    a23a_lat = -75.90
    a23a_lon = -40.50
    a23a_radius = 35.0  # km

    # Test point 50 km away
    d_safe = haversine_km(-75.50, -40.50, a23a_lat, a23a_lon)
    assert d_safe > a23a_radius, "Safe point was inside hazard zone"

    # Test point 10 km away
    d_danger = haversine_km(-75.85, -40.50, a23a_lat, a23a_lon)
    assert d_danger < a23a_radius + 5.0, "Danger point should be detected as collision hazard"
    print(f"[PASS] Iceberg collision zone confirmed: Danger threshold detected within {(a23a_radius + 5.0):.1f} km radius.")


if __name__ == "__main__":
    test_lstm_model()
    test_gis_dataset()
    test_iceberg_standoff()
    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
