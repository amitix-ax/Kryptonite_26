"""
Trajectory export (spec principle L: every output must contain
uncertainty/status metadata where applicable).

Supports CSV (always available) and GeoJSON (requires pyproj for the
projected -> geographic conversion, since we never store lat/lon as the
internal state per spec section 4).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.numerical.deterministic_solver import SimulationResult
from iceberg_model.state.iceberg_state import DynamicsMode


def export_csv(result: SimulationResult, path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "x_m", "y_m", "u_ms", "v_ms", "L_m", "W_m", "H_m", "mode"])
        for seg in result.segments:
            for row, t in zip(seg.states, seg.times):
                x, y, u, v, L, W, H = row
                writer.writerow([t, x, y, u, v, L, W, H, seg.mode.value])


def export_geojson(result: SimulationResult, path: str | Path, config: PhysicsConfig) -> None:
    """
    Convert working-CRS (x, y) to geographic lon/lat for display ONLY,
    using pyproj. Internal state remains projected-meters throughout the
    simulation (spec section 4) -- this conversion happens purely at the
    I/O boundary.
    """
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError("pyproj is required for GeoJSON export.") from e

    transformer = Transformer.from_crs(config.crs.working_crs, config.crs.geographic_crs, always_xy=True)

    features = []
    for seg in result.segments:
        coords = []
        for row in seg.states:
            x, y = row[0], row[1]
            lon, lat = transformer.transform(x, y)
            coords.append([lon, lat])
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "mode": seg.mode.value,
                "t_start": float(seg.times[0]),
                "t_end": float(seg.times[-1]),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "status": result.final_mode.value,
            "events": result.event_log.events,
            "note": (
                "Research/prototype decision-support output. NOT a "
                "certified navigation product; no guarantee of real-world "
                "safety is implied."
            ),
        },
    }

    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)


def export_multi_trajectory_geojson(
    results: dict[str, SimulationResult],
    config: PhysicsConfig,
    path: str | Path,
    route_waypoints_lonlat: list[tuple[float, float]] | None = None,
    route_label: str = "illustrative route",
    stations: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Combine several named SimulationResults (e.g. one per seeded iceberg
    along a corridor) plus an optional route polyline and named station
    markers into a single GeoJSON FeatureCollection, for overlaying a
    whole scenario (route + stations + drift trajectories) on one map.

    `results`: {name: SimulationResult} -- each becomes one or more
        LineString features (one per dynamics-mode segment, same as
        export_geojson), tagged with `properties.iceberg = name`.
    `route_waypoints_lonlat`: optional [(lon, lat), ...] already in
        geographic coordinates -- drawn as-is, NOT reprojected (unlike
        the trajectories, which are converted from the working CRS).
        This is an illustrative interpolation between waypoints, not a
        recorded vessel track, and is labeled as such in its properties.
    `stations`: optional {name: (lon, lat)} rendered as Point features.
    """
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError("pyproj is required for GeoJSON export.") from e

    transformer = Transformer.from_crs(config.crs.working_crs, config.crs.geographic_crs, always_xy=True)
    features = []

    if route_waypoints_lonlat:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [list(p) for p in route_waypoints_lonlat]},
            "properties": {
                "kind": "route",
                "label": route_label,
                "note": "Illustrative interpolation between waypoints, not a recorded vessel track.",
            },
        })

    if stations:
        for name, (lon, lat) in stations.items():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"kind": "station", "name": name},
            })

    for name, result in results.items():
        for seg in result.segments:
            coords = []
            for row in seg.states:
                x, y = row[0], row[1]
                lon, lat = transformer.transform(x, y)
                coords.append([lon, lat])
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "kind": "iceberg_trajectory",
                    "iceberg": name,
                    "mode": seg.mode.value,
                    "t_start": float(seg.times[0]),
                    "t_end": float(seg.times[-1]),
                },
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "icebergs": {
                name: {"final_mode": result.final_mode.value, "events": result.event_log.events}
                for name, result in results.items()
            },
            "note": (
                "Research/prototype decision-support output. NOT a "
                "certified navigation product; no guarantee of real-world "
                "safety is implied. Route geometry is illustrative, not a "
                "recorded vessel track."
            ),
        },
    }

    path = Path(path)
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)


def export_dashboard_json(
    result: SimulationResult,
    config: PhysicsConfig,
    path: str | Path,
    stations: dict[str, tuple[float, float]] | None = None,
    route_waypoints_lonlat: list[tuple[float, float]] | None = None,
    data_sources: dict[str, str] | None = None,
) -> None:
    """Export an enriched JSON file for the interactive frontend dashboard.

    Produces a single file containing everything the dashboard needs:
    - Full state vector time series with lon/lat coordinates
    - Per-timestep uncertainty estimates (position uncertainty growth)
    - Simulation metadata (config, events, data sources)
    - Station locations and route waypoints
    """
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError("pyproj is required for dashboard JSON export.") from e

    transformer = Transformer.from_crs(
        config.crs.working_crs, config.crs.geographic_crs, always_xy=True
    )

    # Build per-step records with lon/lat
    steps = []
    t0 = None
    for seg in result.segments:
        for row, t in zip(seg.states, seg.times):
            if t0 is None:
                t0 = t
            x, y, u, v, L, W, H = row
            lon, lat = transformer.transform(x, y)
            speed = float(np.sqrt(u**2 + v**2))
            heading = float(np.degrees(np.arctan2(u, v))) % 360  # heading from north
            elapsed_h = (t - t0) / 3600.0

            # Position uncertainty grows with sqrt(time) via diffusion σ
            # σ_pos(t) ≈ σ_v * sqrt(elapsed_t) (Brownian motion position spread)
            sigma_v = 0.02  # default diffusion velocity_sigma_ms
            pos_uncertainty_m = sigma_v * np.sqrt(max(t - t0, 0.0))

            volume = L * W * H

            steps.append({
                "t": float(t),
                "elapsed_h": round(elapsed_h, 4),
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "u": round(float(u), 6),
                "v": round(float(v), 6),
                "speed": round(speed, 6),
                "heading": round(heading, 1),
                "L": round(float(L), 2),
                "W": round(float(W), 2),
                "H": round(float(H), 2),
                "volume": round(float(volume), 1),
                "mode": seg.mode.value,
                "pos_uncertainty_m": round(float(pos_uncertainty_m), 1),
            })

    # Compute total displacement
    if len(steps) >= 2:
        dx = steps[-1]["x"] - steps[0]["x"]
        dy = steps[-1]["y"] - steps[0]["y"]
        total_displacement_km = round(np.sqrt(dx**2 + dy**2) / 1000.0, 2)
    else:
        total_displacement_km = 0.0

    # Build default stations if none provided
    DEFAULT_STATIONS = {
        "Maitri (India)": (11.7319, -70.7667),
        "Bharati (India)": (76.1874, -69.4080),
        "Neumayer III (Germany)": (-8.27, -70.67),
        "Troll (Norway)": (2.53, -72.01),
        "Showa (Japan)": (39.58, -69.00),
        "Halley VI (UK)": (-26.21, -75.58),
    }
    stations_map = stations or DEFAULT_STATIONS

    station_features = []
    for name, coords in stations_map.items():
        station_features.append({"name": name, "lon": coords[0], "lat": coords[1]})

    # Default Indian Antarctic Supply Route Corridor
    DEFAULT_ROUTE = [
        (76.1874, -69.4080),  # Bharati
        (70.0468, -67.6020),
        (60.9664, -67.4708),
        (46.8994, -68.4125),
        (25.7989, -69.8250),
        (11.7319, -70.7667),  # Maitri
    ]
    route_coords = route_waypoints_lonlat or DEFAULT_ROUTE
    route = [{"lon": lon, "lat": lat} for lon, lat in route_coords]

    # Supply Ships & Polar Research Vessels
    vessels = [
        {"name": "S.A. Agulhas II (Supply Vessel)", "lon": 15.2, "lat": -68.8, "heading": 240, "status": "In Transit"},
        {"name": "MV Vasiliy Golovnin (Resupply)", "lon": 70.5, "lat": -68.2, "heading": 90, "status": "Anchored"},
        {"name": "RRS Sir David Attenborough", "lon": -10.5, "lat": -69.1, "heading": 110, "status": "Icebreaking Patrol"},
    ]

    # Sea Ice Boundary Zone (Antarctic Coastal Shelf Ice Pack)
    sea_ice_bounds = [
        [-65.0, -30.0], [-66.5, 0.0], [-67.0, 30.0], [-66.0, 60.0],
        [-65.5, 90.0], [-66.0, 120.0], [-67.5, 150.0], [-70.0, 180.0],
        [-70.0, -150.0], [-68.0, -120.0], [-66.0, -90.0], [-64.5, -60.0],
        [-65.0, -30.0]
    ]

    dashboard = {
        "version": 1,
        "metadata": {
            "final_mode": result.final_mode.value,
            "total_steps": len(steps),
            "total_displacement_km": total_displacement_km,
            "duration_hours": round(steps[-1]["elapsed_h"], 2) if steps else 0,
            "events": result.event_log.events,
            "data_sources": data_sources or {},
            "config": {
                "working_crs": config.crs.working_crs,
                "geographic_crs": config.crs.geographic_crs,
                "sea_ice_drag_enabled": config.sea_ice_drag.enabled,
                "grounding_enabled": config.grounding.enabled,
                "melting_enabled": config.melting.enabled,
                "solver": config.solver.kind.value if hasattr(config.solver.kind, "value") else str(config.solver.kind),
            },
            "note": (
                "Research/prototype decision-support output. NOT a "
                "certified navigation product."
            ),
        },
        "steps": steps,
        "stations": station_features,
        "route": route,
        "vessels": vessels,
        "sea_ice_bounds": sea_ice_bounds,
    }

    path = Path(path)
    with open(path, "w") as f:
        json.dump(dashboard, f, indent=2)


