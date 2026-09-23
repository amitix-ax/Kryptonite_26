"""
Iceberg Trajectory Predictor
=============================
Physics-driven + ML hybrid predictor for small iceberg drift trajectories.

Physics model based on:
  - Bigg et al. (1997) iceberg drift equations
  - Gladstone et al. (2001) drag parametrization  
  - Smith & Banke (1983) wave radiation force

Force balance on a free-floating iceberg:
  M * dV/dt = F_wind + F_current + F_coriolis + F_wave + F_sea_ice - F_skin_drag

Small icebergs (< 200m) are more responsive to currents than wind because:
  - Larger sail-area-to-draft ratio for large bergs makes them wind-driven
  - Small bergs are mostly submerged, so ocean drag dominates
  - Wave radiation force is significant for bergs with L comparable to wavelength

This module:
1. Propagates iceberg position using force-balance physics
2. Applies the trained LSTM model for residual correction
3. Generates uncertainty cones for multi-step forecasts
4. Handles melt-rate estimation (reduces iceberg size over time)
"""

import math
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from environment_simulator import get_field_at_point, GRID_H, GRID_W, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
except ImportError:
    pass

# Physical constants
RHO_AIR = 1.225       # kg/m³ — air density at sea level
RHO_WATER = 1025.0    # kg/m³ — seawater density
RHO_ICE = 917.0       # kg/m³ — ice density
C_DA = 0.5            # air drag coefficient (Bigg et al.)
C_DW = 0.9            # water drag coefficient (Gladstone et al.)
OMEGA = 7.292e-5      # Earth rotation rate (rad/s)
G = 9.81              # gravitational acceleration (m/s²)

# Melt rate coefficients
MELT_BASAL = 0.58     # m/day per °C above freezing (Bigg et al. 1997)
MELT_WAVE = 0.02      # m/day per m wave height (Gladstone et al. 2001)


def compute_forces(
    iceberg: Dict,
    env_state: Dict,
    lat: float,
    lon: float,
) -> Tuple[float, float]:
    """
    Compute net force per unit mass (acceleration) on an iceberg.

    Returns (ax, ay) in m/s² where x=east, y=north.
    """
    L = iceberg.get("length_m", 50.0)
    W = iceberg.get("width_m", L * 0.6)
    H = iceberg.get("height_m", 10.0)
    draft = iceberg.get("draft_m", H * 5.0)
    mass = RHO_ICE * L * W * (H + draft)

    u_berg = iceberg.get("u", 0.0)
    v_berg = iceberg.get("v", 0.0)

    # Get environmental fields at iceberg position
    try:
        wind_u = get_field_at_point(env_state, lat, lon, 'wind_u')
        wind_v = get_field_at_point(env_state, lat, lon, 'wind_v')
        ocean_u = get_field_at_point(env_state, lat, lon, 'ocean_u')
        ocean_v = get_field_at_point(env_state, lat, lon, 'ocean_v')
        ice_conc = get_field_at_point(env_state, lat, lon, 'ice_conc')
        wave_ht = get_field_at_point(env_state, lat, lon, 'wave_height')
    except Exception:
        wind_u, wind_v = 10.0, -2.0
        ocean_u, ocean_v = 0.1, -0.03
        ice_conc = 0.1
        wave_ht = 2.0

    # ── Wind Drag ──────────────────────────────────────────
    sail_area = L * H  # freeboard face
    rel_wind_u = wind_u - u_berg
    rel_wind_v = wind_v - v_berg
    wind_speed_rel = math.sqrt(rel_wind_u**2 + rel_wind_v**2) + 1e-8
    F_wind_u = 0.5 * RHO_AIR * C_DA * sail_area * wind_speed_rel * rel_wind_u
    F_wind_v = 0.5 * RHO_AIR * C_DA * sail_area * wind_speed_rel * rel_wind_v

    # ── Ocean Current Drag ──────────────────────────────────
    keel_area = L * draft  # underwater face
    rel_curr_u = ocean_u - u_berg
    rel_curr_v = ocean_v - v_berg
    curr_speed_rel = math.sqrt(rel_curr_u**2 + rel_curr_v**2) + 1e-8
    F_curr_u = 0.5 * RHO_WATER * C_DW * keel_area * curr_speed_rel * rel_curr_u
    F_curr_v = 0.5 * RHO_WATER * C_DW * keel_area * curr_speed_rel * rel_curr_v

    # ── Coriolis Force ─────────────────────────────────────
    f = 2 * OMEGA * math.sin(math.radians(lat))
    F_cor_u = mass * f * v_berg
    F_cor_v = -mass * f * u_berg

    # ── Wave Radiation Force ───────────────────────────────
    # Force proportional to wave height² and iceberg cross-section
    if wave_ht > 0.1:
        F_wave_mag = 0.5 * RHO_WATER * G * wave_ht**2 * W * 0.5
        # Wave force acts in the dominant wave propagation direction (assume wind-driven)
        wind_dir = math.atan2(wind_v, wind_u + 1e-8)
        F_wave_u = F_wave_mag * math.cos(wind_dir)
        F_wave_v = F_wave_mag * math.sin(wind_dir)
    else:
        F_wave_u, F_wave_v = 0.0, 0.0

    # ── Sea Ice Resistance ─────────────────────────────────
    if ice_conc > 0.15:
        ice_resist_factor = ice_conc ** 1.5
        F_ice_u = -ice_resist_factor * mass * u_berg * 0.001
        F_ice_v = -ice_resist_factor * mass * v_berg * 0.001
    else:
        F_ice_u, F_ice_v = 0.0, 0.0

    # ── Net Acceleration ───────────────────────────────────
    Fx = F_wind_u + F_curr_u + F_cor_u + F_wave_u + F_ice_u
    Fy = F_wind_v + F_curr_v + F_cor_v + F_wave_v + F_ice_v

    ax = Fx / mass
    ay = Fy / mass

    return ax, ay


def compute_melt_rate(
    iceberg: Dict,
    env_state: Dict,
    lat: float,
    lon: float,
) -> Dict[str, float]:
    """
    Compute daily melt rates (m/day) for basal, wave erosion, and forced convection.
    """
    try:
        sst = get_field_at_point(env_state, lat, lon, 'sst')
        wave_ht = get_field_at_point(env_state, lat, lon, 'wave_height')
        ocean_u = get_field_at_point(env_state, lat, lon, 'ocean_u')
        ocean_v = get_field_at_point(env_state, lat, lon, 'ocean_v')
    except Exception:
        sst, wave_ht, ocean_u, ocean_v = 0.0, 1.5, 0.1, 0.0

    u_berg = iceberg.get("u", 0.0)
    v_berg = iceberg.get("v", 0.0)

    # Temperature above freezing (-1.8°C)
    delta_T = max(0, sst - (-1.8))

    # Basal melt: proportional to ΔT
    basal_melt = MELT_BASAL * delta_T

    # Wave erosion: proportional to wave height
    wave_melt = MELT_WAVE * wave_ht

    # Forced convection: proportional to relative water speed
    rel_speed = math.sqrt((ocean_u - u_berg)**2 + (ocean_v - v_berg)**2)
    forced_melt = 0.15 * delta_T * rel_speed ** 0.8

    total_melt = basal_melt + wave_melt + forced_melt

    return {
        "basal_melt_m_day": round(basal_melt, 4),
        "wave_melt_m_day": round(wave_melt, 4),
        "forced_convection_m_day": round(forced_melt, 4),
        "total_melt_m_day": round(total_melt, 4),
    }


def predict_trajectory(
    iceberg: Dict,
    env_states: List[Dict],
    dt_hours: float = 1.0,
    forecast_hours: float = 72.0,
    use_lstm_correction: bool = True,
) -> List[Dict]:
    """
    Predict iceberg trajectory using physics + optional LSTM residual correction.

    Args:
        iceberg: Iceberg dict with lat, lon, u, v, physical dimensions.
        env_states: List of environmental state snapshots (time-ordered).
        dt_hours: Timestep in hours.
        forecast_hours: Total forecast horizon.
        use_lstm_correction: Whether to apply LSTM residual correction.

    Returns:
        List of trajectory points with position, velocity, uncertainty.
    """
    dt = dt_hours * 3600.0  # seconds
    n_steps = int(forecast_hours / dt_hours)

    # Current state
    lat = iceberg["lat"]
    lon = iceberg["lon"]
    u = iceberg.get("u", 0.0)
    v = iceberg.get("v", 0.0)
    L = iceberg.get("length_m", 50.0)
    W = iceberg.get("width_m", L * 0.6)
    H = iceberg.get("height_m", 10.0)
    draft = iceberg.get("draft_m", H * 5.0)

    # LSTM model for residual correction
    lstm_model = None
    if use_lstm_correction:
        try:
            import torch
            from iceberg_lstm import IcebergDriftLSTM
            import os
            ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints_local", "iceberg_lstm.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                lstm_model = IcebergDriftLSTM(
                    input_dim=ckpt.get("input_dim", 6),
                    hidden_dim=ckpt.get("hidden_dim", 64),
                    num_layers=ckpt.get("num_layers", 2),
                    output_dim=ckpt.get("output_dim", 4),
                )
                lstm_model.load_state_dict(ckpt["model_state_dict"])
                lstm_model.eval()
        except Exception as e:
            lstm_model = None

    trajectory = [{
        "step": 0,
        "elapsed_h": 0.0,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "u": round(u, 5),
        "v": round(v, 5),
        "speed_ms": round(math.sqrt(u**2 + v**2), 5),
        "heading_deg": round((math.degrees(math.atan2(v, u)) + 360) % 360, 1),
        "length_m": round(L, 1),
        "height_m": round(H, 1),
        "uncertainty_km": 0.0,
        "melt_info": {},
    }]

    # Rolling history for LSTM
    history = [{"lat": lat, "lon": lon, "u": u, "v": v, "speed": math.sqrt(u**2 + v**2), "heading": 0.0}]

    for step_idx in range(1, n_steps + 1):
        # Select environment state (use nearest available)
        env_idx = min(step_idx - 1, len(env_states) - 1) if env_states else 0
        env = env_states[env_idx] if env_states else {}

        # Working copy of iceberg state
        berg_state = {
            "length_m": L, "width_m": W, "height_m": H, "draft_m": draft,
            "u": u, "v": v,
        }

        # Physics integration (RK2 midpoint method)
        ax1, ay1 = compute_forces(berg_state, env, lat, lon)
        mid_u = u + ax1 * dt * 0.5
        mid_v = v + ay1 * dt * 0.5
        mid_lat = lat + (mid_v / 111320.0) * dt * 0.5
        mid_lon = lon + (mid_u / (111320.0 * math.cos(math.radians(lat)))) * dt * 0.5
        berg_state["u"] = mid_u
        berg_state["v"] = mid_v
        ax2, ay2 = compute_forces(berg_state, env, mid_lat, mid_lon)

        u += ax2 * dt
        v += ay2 * dt
        lat += (v / 111320.0) * dt
        lon += (u / (111320.0 * math.cos(math.radians(lat + 1e-6)))) * dt

        # LSTM residual correction (if available)
        if lstm_model is not None and len(history) >= 4:
            try:
                import torch
                from iceberg_lstm import extract_step_features
                ref_lat, ref_lon = history[0]["lat"], history[0]["lon"]
                seq = [extract_step_features(h, ref_lat, ref_lon) for h in history[-6:]]
                seq_tensor = torch.tensor([seq], dtype=torch.float32)
                with torch.no_grad():
                    preds, sigma = lstm_model(seq_tensor)
                    correction = preds[0].numpy()
                    # Apply small correction factor (residual learning)
                    lat += float(correction[0]) * 0.1
                    lon += float(correction[1]) * 0.1
            except Exception:
                pass

        # Clip to domain
        lat = max(LAT_MIN, min(LAT_MAX, lat))
        lon = max(LON_MIN, min(LON_MAX, lon))

        # Speed limit (icebergs rarely exceed 1 m/s)
        speed = math.sqrt(u**2 + v**2)
        if speed > 1.0:
            u *= 1.0 / speed
            v *= 1.0 / speed
            speed = 1.0

        # Melt rate
        melt = compute_melt_rate(berg_state, env, lat, lon)
        daily_melt = melt["total_melt_m_day"]
        hourly_melt = daily_melt / 24.0
        H = max(0.5, H - hourly_melt * dt_hours)
        L = max(1.0, L - hourly_melt * 0.5 * dt_hours)
        W = max(0.5, W - hourly_melt * 0.3 * dt_hours)
        draft = H * 5.0

        heading = (math.degrees(math.atan2(v, u)) + 360) % 360

        # Uncertainty grows with time (ensemble spread model)
        uncertainty_km = 0.5 * math.sqrt(step_idx * dt_hours) * (1.0 + speed * 2.0)

        trajectory.append({
            "step": step_idx,
            "elapsed_h": round(step_idx * dt_hours, 2),
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "u": round(u, 5),
            "v": round(v, 5),
            "speed_ms": round(speed, 5),
            "heading_deg": round(heading, 1),
            "length_m": round(L, 1),
            "height_m": round(H, 1),
            "uncertainty_km": round(uncertainty_km, 2),
            "melt_info": melt,
        })

        history.append({
            "lat": lat, "lon": lon, "u": u, "v": v,
            "speed": speed, "heading": heading,
        })

    return trajectory


def batch_predict_trajectories(
    icebergs: List[Dict],
    env_states: List[Dict],
    forecast_hours: float = 48.0,
    dt_hours: float = 1.0,
) -> Dict[str, List[Dict]]:
    """Predict trajectories for all icebergs in batch."""
    results = {}
    for berg in icebergs:
        berg_id = berg.get("id", "unknown")
        try:
            traj = predict_trajectory(berg, env_states, dt_hours, forecast_hours)
            results[berg_id] = traj
        except Exception as e:
            results[berg_id] = [{"error": str(e)}]
    return results


if __name__ == "__main__":
    from environment_simulator import AntarcticEnvironment
    from small_iceberg_detector import generate_iceberg_population

    print("=" * 60)
    print("Iceberg Trajectory Predictor — Test Run")
    print("=" * 60)

    env = AntarcticEnvironment(seed=42)
    start_time = datetime.datetime(2026, 3, 15, 0, 0)

    # Generate 6-hourly environment states for 48 hours
    env_states = env.generate_timeseries(start_time, hours=48, dt_hours=6.0)
    print(f"Generated {len(env_states)} environment snapshots")

    # Generate 5 test icebergs
    icebergs = generate_iceberg_population(5, timestamp=start_time, seed=99, env_state=env_states[0])

    for berg in icebergs:
        print(f"\n--- {berg['id']} ({berg['size_class']}, L={berg['length_m']}m) ---")
        print(f"  Start: ({berg['lat']:.3f}, {berg['lon']:.3f})")

        traj = predict_trajectory(berg, env_states, dt_hours=3.0, forecast_hours=48.0)
        last = traj[-1]
        print(f"  End:   ({last['lat']:.3f}, {last['lon']:.3f})")
        print(f"  Displacement: ~{math.sqrt((last['lat']-berg['lat'])**2 + (last['lon']-berg['lon'])**2) * 111:.1f} km")
        print(f"  Final speed: {last['speed_ms']:.4f} m/s, heading: {last['heading_deg']:.1f}°")
        print(f"  Final size: L={last['length_m']:.1f}m, H={last['height_m']:.1f}m")
        print(f"  Uncertainty cone: ±{last['uncertainty_km']:.1f} km")
