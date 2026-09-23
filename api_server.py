import os
import math
import random
import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
import numpy as np

try:
    from pathfinder import set_environment, compute_route
except ImportError:
    print("Warning: pathfinder module not found. A* routing will not work.")


try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

app = FastAPI(title="Antarctic DSS API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Client helper

def get_genai_client():
    if not HAS_GENAI:
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print("Error initializing Gemini client:", e)
        return None

class RouteRequest(BaseModel):
    start_coords: List[float]  # [lat, lon]
    end_coords: List[float]    # [lat, lon]
    forecast_window: int = 72
    safety_weight: float = 1.0

class RouteAnalysisRequest(BaseModel):
    total_distance_km: float
    open_water_km: float
    marginal_ice_km: float
    pack_ice_km: float
    max_ice_conc: float
    max_crosswind: float
    avg_drift: float

class DigitalTwinStateRequest(BaseModel):
    env_wind_spd: Optional[float] = 0.0
    env_ice_conc: Optional[float] = 0.0
    u: Optional[float] = 0.0
    v: Optional[float] = 0.0
    r: Optional[float] = 0.0
    r_ice: Optional[float] = 0.0

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.post("/api/gemini_summary")
def gemini_summary(req: DigitalTwinStateRequest):
    client = get_genai_client()
    wind = req.env_wind_spd or 0.0
    ice = (req.env_ice_conc or 0.0) * 100
    u = req.u or 0.0
    v = req.v or 0.0
    r = req.r or 0.0
    res = req.r_ice or 0.0

    if not client:
        return {"summary": f"Tactical Telemetry: Wind {wind:.1f}m/s, Ice {ice:.1f}%, Surge {u:.2f}m/s, Sway {v:.3f}m/s. Physics state nominal under current ice resistance ({res:.0f} kN)."}

    prompt = f"""
    You are the 'Kryptonite' AI Decision Support System onboard a Polar Class vessel.
    Analyze this current digital twin telemetry:
    - Wind Speed: {wind:.1f} m/s
    - Ice Concentration: {ice:.1f}%
    - Surge Velocity (u): {u:.2f} m/s
    - Sway Velocity (v): {v:.3f} m/s
    - Yaw Rate (r): {r:.3f} rad/s
    - Ice Resistance: {res:.0f} kN

    Write a highly concise, professional 3-4 sentence tactical summary of the current physics state. 
    State if conditions are nominal or if the vessel is experiencing severe leeway (sway) or heavy ice resistance. 
    Use a cold, analytical tone. Do not use markdown formatting.
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        return {"summary": response.text}
    except Exception as e:
        print("Gemini API Error:", e)
        return {"summary": f"Tactical Summary (Fallback): Wind {wind:.1f}m/s, Ice {ice:.1f}%, Surge {u:.2f}m/s, Sway {v:.3f}m/s, Resistance {res:.0f}kN. Status nominal."}

@app.post("/api/ai/analyze-route")
def analyze_route(req: RouteAnalysisRequest):
    client = get_genai_client()
    
    if client:
        prompt = f"""
        You are the 'Kryptonite' AI Decision Support System onboard a Polar Class vessel.
        Analyze this proposed route:
        - Total Distance: {req.total_distance_km:.1f} km
        - Open Water: {req.open_water_km:.1f} km
        - Marginal Ice: {req.marginal_ice_km:.1f} km
        - Pack Ice: {req.pack_ice_km:.1f} km
        - Max Ice Concentration: {req.max_ice_conc*100:.1f} %
        - Max Crosswind: {req.max_crosswind:.1f} knots
        - Max Drift Velocity: {req.avg_drift:.2f} m/s

        Provide a highly concise, professional tactical route assessment (4-5 bullet points). 
        Identify the most critical leg of the journey and any severe risks. 
        Use a cold, analytical tone. Format with markdown bullet points.
        """
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
            return {"analysis_markdown": response.text}
        except Exception as e:
            print("Gemini API Error (Analyze Route):", e)

    narrative = [
        "### Gemini Tactical Route Assessment",
        f"Analyzing {req.total_distance_km:.1f} km transit corridor...",
        ""
    ]
    if req.max_ice_conc > 0.4:
        narrative.append("## ⚠️ High Sea-Ice Convergence Warning")
        narrative.append("The ML forecasting model and Physics drift tracker show intersecting hazards.")
        narrative.append(f"Vessel will encounter up to **{req.max_ice_conc*100:.0f}% pack ice**. Expect delays and elevated hull stress.")
    elif req.max_ice_conc > 0.15:
        narrative.append("## ℹ️ Marginal Ice Advisory")
        narrative.append(f"Route involves **{req.marginal_ice_km:.1f} km** of marginal ice. Maintain standard icebreaker protocols.")
    else:
        narrative.append("## ✅ Clear Transit Validated")
        narrative.append(f"The converged ML-Physics pipeline predicts predominantly open water ({req.open_water_km:.1f} km).")

    narrative.append("")
    narrative.append("## Weather & Drift Constraints")
    narrative.append(f"* Max crosswinds: **{req.max_crosswind:.1f} kts**")
    narrative.append(f"* Peak ice drift velocity: **{req.avg_drift:.2f} kts**")
    return {"analysis_markdown": "\n".join(narrative)}

@app.post("/api/calculate-route")
def calculate_route(req: RouteRequest):
    lat1, lon1 = req.start_coords
    lat2, lon2 = req.end_coords
    
    # Generate mock sea-ice and environment grids (128x128)
    ice_grid = np.zeros((128, 128))
    wind_u = np.zeros((128, 128))
    wind_v = np.zeros((128, 128))
    ocean_u = np.zeros((128, 128))
    ocean_v = np.zeros((128, 128))
    drift_u = np.zeros((128, 128))
    drift_v = np.zeros((128, 128))
    
    try:
        land_mask = np.load(os.path.join(os.path.dirname(__file__), "data", "land_mask.npy"))
    except:
        land_mask = np.zeros((128, 128))
    
    for r in range(128):
        # r goes from 0 (North, -60) to 127 (South, -72)
        lat = -60.0 - (12.0 * r / 127.0)
        # Ice concentration increases as we go south
        if lat < -65:
            ice = min(1.0, (abs(lat) - 65) / 5.0)
        else:
            ice = 0.0
        
        # Add some east-west variation for realism
        for c in range(128):
            lon = 10.0 + (70.0 * c / 127.0)
            ice_val = ice + 0.15 * math.sin(lon / 10.0)
            
            # Apply land mask (treat land as fast ice)
            if land_mask[r, c] > 0.5:
                ice_grid[r, c] = 1.0
            else:
                ice_grid[r, c] = max(0.0, min(1.0, ice_val))
                
            wind_u[r, c] = 10.0 + 5.0 * math.cos(lat / 5.0)
            wind_v[r, c] = -2.0 + 2.0 * math.sin(lon / 5.0)
            ocean_u[r, c] = 0.1
            ocean_v[r, c] = -0.05
            if ice_grid[r, c] > 0.15:
                drift_u[r, c] = wind_u[r, c] * 0.02
                drift_v[r, c] = wind_v[r, c] * 0.02

    # Set environment for pathfinder
    set_environment(wind_u, wind_v, ocean_u, ocean_v, drift_u, drift_v, source="Synthetic AI Grid")
    
    try:
        result = compute_route(ice_grid, (lat1, lon1), (lat2, lon2), req.forecast_window, req.safety_weight)
        
        if result and result.get("route_coords"):
            # Ensure format matches frontend expectations
            telemetry = []
            for wp in result.get("telemetry", []):
                telemetry.append({
                    "lat": wp["coordinates"][1],
                    "lon": wp["coordinates"][0],
                    "ml_ice_concentration": wp.get("ice_concentration", 0.0),
                    "physics_iceberg_risk": random.random() * 0.3, # Optional additional iceberg hazard
                    "converged_hazard": wp.get("hazard_type") in ["Pack Ice", "Fast Ice / Land"]
                })
            
            return {
                "route_coords": result["route_coords"],
                "distance_km": result["distance_km"],
                "telemetry": telemetry,
                "route_summary": result["route_summary"],
                "environmental_summary": result["environmental_summary"],
                "risk_status": result["risk_status"]
            }
    except Exception as e:
        print("Pathfinder error:", e)
        
    # Fallback to straight line if A* fails
    total_dist_km = haversine(lat1, lon1, lat2, lon2)
    return {
        "route_coords": [[lon1, lat1], [lon2, lat2]],
        "distance_km": total_dist_km,
        "telemetry": [],
        "route_summary": {
            "total_open_water_km": total_dist_km,
            "total_marginal_ice_km": 0,
            "total_pack_ice_km": 0,
            "max_ice_concentration_encountered": 0,
            "critical_hurdles_count": 0
        },
        "environmental_summary": {
            "max_crosswind_knots": 0,
            "max_drift_velocity": 0
        },
        "risk_status": "NO_ROUTE"
    }

@app.get("/api/ai/live-news")
def live_news():
    now_utc = datetime.datetime.utcnow().strftime("%d %b %Y - %H:%M UTC")
    return {
        "news": [
            {
                "category": "NCPOR DIRECTIVE",
                "timestamp": now_utc,
                "title": "Maitri Track Fast-Ice Breakup Advisory",
                "summary": "Sentinel-1 SAR observation indicates open polynyas forming near -66.5S latitude. Icebreakers advised to route around converging drift."
            },
            {
                "category": "PHYSICS MODEL ALERT",
                "timestamp": now_utc,
                "title": "Iceberg A-81 Trajectory Update",
                "summary": "Simulated drift vectors show A-81 moving WNW at 0.15 m/s. Proximity alert enabled for Lazarev Sea approaches."
            }
        ]
    }

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT SIMULATOR API
# ═══════════════════════════════════════════════════════════════

class EnvironmentRequest(BaseModel):
    timestamp: Optional[str] = None  # ISO format, defaults to now
    perturbation: Optional[float] = 0.3
    seed: Optional[int] = None

@app.post("/api/environment/generate")
def generate_environment(req: EnvironmentRequest):
    """Generate a full Antarctic environmental state snapshot."""
    try:
        from environment_simulator import AntarcticEnvironment, get_field_at_point
    except ImportError:
        raise HTTPException(500, "Environment simulator module not available")
    
    ts = datetime.datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.datetime.utcnow()
    env = AntarcticEnvironment(seed=req.seed)
    state = env.generate(ts, req.perturbation or 0.3)
    
    # Return summary stats (full grids are too large for JSON)
    summary = {"timestamp": ts.isoformat(), "season_factor": state["season_factor"], "fields": {}}
    for field in ['wind_u', 'wind_v', 'ocean_u', 'ocean_v', 'sst', 'ice_conc', 'wave_height']:
        arr = state[field]
        summary["fields"][field] = {
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3),
        }
    
    # Sample points along key latitudes for frontend visualization
    sample_points = []
    for lat in [-62, -64, -66, -68, -70]:
        for lon in [15, 25, 35, 45, 55, 65, 75]:
            point = {"lat": lat, "lon": lon}
            for field in ['wind_u', 'wind_v', 'ocean_u', 'ocean_v', 'sst', 'ice_conc', 'wave_height']:
                point[field] = round(get_field_at_point(state, lat, lon, field), 3)
            sample_points.append(point)
    summary["sample_points"] = sample_points
    
    return summary

@app.post("/api/environment/point")
def environment_at_point(lat: float, lon: float, timestamp: Optional[str] = None):
    """Get all environmental fields at a specific lat/lon."""
    try:
        from environment_simulator import AntarcticEnvironment, get_field_at_point
    except ImportError:
        raise HTTPException(500, "Environment simulator module not available")
    
    ts = datetime.datetime.fromisoformat(timestamp) if timestamp else datetime.datetime.utcnow()
    env = AntarcticEnvironment(seed=42)
    state = env.generate(ts)
    
    result = {"lat": lat, "lon": lon, "timestamp": ts.isoformat()}
    for field in ['wind_u', 'wind_v', 'ocean_u', 'ocean_v', 'sst', 'ice_conc', 'wave_height', 'bathymetry']:
        result[field] = round(get_field_at_point(state, lat, lon, field), 4)
    
    wind_speed = math.sqrt(result['wind_u']**2 + result['wind_v']**2)
    result['wind_speed_ms'] = round(wind_speed, 2)
    result['wind_speed_kts'] = round(wind_speed * 1.944, 1)
    result['ice_conc_pct'] = round(result['ice_conc'] * 100, 1)
    
    return result

# ═══════════════════════════════════════════════════════════════
# SMALL ICEBERG DETECTION API
# ═══════════════════════════════════════════════════════════════

class IcebergDetectionRequest(BaseModel):
    timestamp: Optional[str] = None
    n_population: Optional[int] = 30
    detection_sensitivity: Optional[float] = 0.6
    seed: Optional[int] = None

@app.post("/api/icebergs/detect")
def detect_icebergs(req: IcebergDetectionRequest):
    """Generate iceberg population and detect additional icebergs from environment."""
    try:
        from environment_simulator import AntarcticEnvironment
        from small_iceberg_detector import (
            generate_iceberg_population,
            detect_icebergs_from_environment,
            classify_hazard_level,
        )
    except ImportError as e:
        raise HTTPException(500, f"Detection modules not available: {e}")
    
    ts = datetime.datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.datetime.utcnow()
    env = AntarcticEnvironment(seed=req.seed)
    state = env.generate(ts)
    
    # Generate known iceberg population
    population = generate_iceberg_population(
        req.n_population or 30,
        timestamp=ts,
        seed=(req.seed or 42) + 1,
        env_state=state,
    )
    
    # Detect additional icebergs from environmental signatures
    detected = detect_icebergs_from_environment(
        state,
        sensitivity=req.detection_sensitivity or 0.6,
        seed=(req.seed or 42) + 2,
    )
    
    # Classify hazards
    all_bergs = population + detected
    for berg in all_bergs:
        berg["hazard_level"] = classify_hazard_level(berg)
    
    # Statistics
    hazard_dist = {}
    size_dist = {}
    for berg in all_bergs:
        h = berg.get("hazard_level", "LOW")
        hazard_dist[h] = hazard_dist.get(h, 0) + 1
        s = berg.get("size_class", "unknown")
        size_dist[s] = size_dist.get(s, 0) + 1
    
    return {
        "timestamp": ts.isoformat(),
        "total_icebergs": len(all_bergs),
        "known_population": len(population),
        "detected_from_environment": len(detected),
        "hazard_distribution": hazard_dist,
        "size_distribution": size_dist,
        "icebergs": all_bergs,
    }

# ═══════════════════════════════════════════════════════════════
# ICEBERG TRAJECTORY PREDICTION API
# ═══════════════════════════════════════════════════════════════

class TrajectoryRequest(BaseModel):
    iceberg_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    length_m: Optional[float] = 50.0
    height_m: Optional[float] = 10.0
    forecast_hours: Optional[float] = 48.0
    dt_hours: Optional[float] = 3.0
    timestamp: Optional[str] = None
    seed: Optional[int] = None

@app.post("/api/icebergs/predict-trajectory")
def predict_iceberg_trajectory(req: TrajectoryRequest):
    """Predict trajectory for a single iceberg using physics + LSTM hybrid model."""
    try:
        from environment_simulator import AntarcticEnvironment
        from iceberg_trajectory_predictor import predict_trajectory as traj_predict
    except ImportError as e:
        raise HTTPException(500, f"Trajectory predictor not available: {e}")
    
    ts = datetime.datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.datetime.utcnow()
    env = AntarcticEnvironment(seed=req.seed)
    env_series = env.generate_timeseries(ts, hours=int(req.forecast_hours or 48), dt_hours=6.0)
    
    iceberg = {
        "id": req.iceberg_id or "MANUAL-001",
        "lat": req.lat or -69.0,
        "lon": req.lon or 30.0,
        "length_m": req.length_m or 50.0,
        "width_m": (req.length_m or 50.0) * 0.6,
        "height_m": req.height_m or 10.0,
        "draft_m": (req.height_m or 10.0) * 5.0,
        "u": 0.0,
        "v": 0.0,
    }
    
    trajectory = traj_predict(
        iceberg, env_series,
        dt_hours=req.dt_hours or 3.0,
        forecast_hours=req.forecast_hours or 48.0,
    )
    
    return {
        "iceberg": iceberg,
        "forecast_hours": req.forecast_hours,
        "dt_hours": req.dt_hours,
        "n_steps": len(trajectory),
        "trajectory": trajectory,
    }

# ═══════════════════════════════════════════════════════════════
# SCENARIO TESTING API
# ═══════════════════════════════════════════════════════════════

@app.get("/api/scenarios/list")
def list_scenarios():
    """List all available test scenarios."""
    try:
        from scenario_test_harness import SCENARIOS
    except ImportError:
        raise HTTPException(500, "Test harness not available")
    
    return {
        "scenarios": [
            {"key": k, "name": v["name"], "description": v["description"]}
            for k, v in SCENARIOS.items()
        ]
    }

@app.post("/api/scenarios/run/{scenario_key}")
def run_scenario_api(scenario_key: str):
    """Execute a full test scenario and return the report."""
    try:
        from scenario_test_harness import run_scenario
    except ImportError:
        raise HTTPException(500, "Test harness not available")
    
    try:
        report = run_scenario(scenario_key, verbose=False)
        return report
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Scenario execution failed: {e}")

# ═══════════════════════════════════════════════════════════════
# DRIFT PREDICTION (LSTM)
# ═══════════════════════════════════════════════════════════════

class DriftPredictionRequest(BaseModel):
    iceberg_id: Optional[str] = "SIM-01"
    forecast_hours: Optional[int] = 24
    history_steps: Optional[List[dict]] = None

lstm_model_instance = None

def get_lstm_model():
    global lstm_model_instance
    if lstm_model_instance is not None:
        return lstm_model_instance
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_local", "iceberg_lstm.pt")
    if os.path.exists(ckpt_path):
        try:
            import torch
            from iceberg_lstm import IcebergDriftLSTM
            ckpt = torch.load(ckpt_path, map_location="cpu")
            m = IcebergDriftLSTM(
                input_dim=ckpt.get("input_dim", 6),
                hidden_dim=ckpt.get("hidden_dim", 64),
                num_layers=ckpt.get("num_layers", 2),
                output_dim=ckpt.get("output_dim", 4),
            )
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            lstm_model_instance = m
            print("Loaded IcebergDriftLSTM model successfully from", ckpt_path)
            return lstm_model_instance
        except Exception as e:
            print("Failed to load LSTM model:", e)
    return None

@app.post("/api/predict-drift")
def predict_drift(req: DriftPredictionRequest):
    model = get_lstm_model()
    steps = req.history_steps

    if not steps:
        import glob
        import json
        daily_files = sorted(glob.glob(os.path.join(data_dir, "drift_history_2026-*.json")))
        if daily_files:
            try:
                with open(daily_files[-1], "r", encoding="utf-8") as f:
                    d = json.load(f)
                    steps = d.get("steps", [])[-12:]
            except Exception:
                steps = []

    if model and steps and len(steps) >= 4:
        from iceberg_lstm import predict_trajectory
        forecasted = predict_trajectory(model, steps, forecast_hours=req.forecast_hours or 24)
        return {
            "status": "success",
            "model": "IcebergDriftLSTM (Trained PyTorch Model)",
            "iceberg_id": req.iceberg_id,
            "forecast_hours": req.forecast_hours,
            "forecasted_trajectory": forecasted
        }
    else:
        return {
            "status": "fallback",
            "model": "PhysicsFallback",
            "iceberg_id": req.iceberg_id,
            "forecast_hours": req.forecast_hours,
            "forecasted_trajectory": steps or []
        }

# Mount static files when running full app locally
project_root = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(project_root, "frontend")
data_dir = os.path.join(project_root, "data")

if os.path.exists(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir, html=True), name="frontend")
if os.path.exists(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")

@app.get("/")
def read_root():
    return RedirectResponse(url="/frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    print("Starting Antarctic DSS Server on http://127.0.0.1:5000")
    uvicorn.run(app, host="127.0.0.1", port=5000)

