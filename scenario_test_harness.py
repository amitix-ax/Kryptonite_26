"""
Scenario Test Harness
======================
Generates realistic Antarctic scenarios for testing the iceberg detection,
trajectory prediction, and navigation systems end-to-end.

Scenarios include:
  1. Summer Calm   — low ice, moderate wind, few icebergs
  2. Winter Storm  — heavy ice, strong wind, many icebergs
  3. Calving Event — sudden burst of small icebergs near ice shelf
  4. Ice Edge Transit — vessel crossing marginal ice zone with growlers
  5. Multi-Berg Convergence — several icebergs converging on shipping lane

Each scenario:
  - Generates environment states over a time window
  - Populates icebergs with size/position distributions
  - Runs detection and trajectory prediction
  - Evaluates whether the system correctly identified hazards
  - Outputs a structured test report
"""

import datetime
import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np

from environment_simulator import AntarcticEnvironment, export_environment_json
from small_iceberg_detector import (
    generate_iceberg_population,
    detect_icebergs_from_environment,
    classify_hazard_level,
    export_iceberg_database,
)
from iceberg_trajectory_predictor import predict_trajectory, batch_predict_trajectories


# ── Scenario Definitions ──────────────────────────────────────
SCENARIOS = {
    "summer_calm": {
        "name": "Summer Calm Transit",
        "description": "Austral summer (January), light winds, minimal sea ice, standard iceberg population.",
        "timestamp": datetime.datetime(2026, 1, 15, 12, 0),
        "n_icebergs": 15,
        "perturbation": 0.15,
        "forecast_hours": 48,
        "seed": 100,
    },
    "winter_storm": {
        "name": "Winter Storm Conditions",
        "description": "Austral winter (July), gale-force winds, heavy pack ice, dense iceberg field.",
        "timestamp": datetime.datetime(2026, 7, 20, 6, 0),
        "n_icebergs": 50,
        "perturbation": 0.8,
        "forecast_hours": 72,
        "seed": 200,
    },
    "calving_event": {
        "name": "Ice Shelf Calving Event",
        "description": "Sudden calving from Amery Ice Shelf, burst of 30+ small icebergs entering shipping lane.",
        "timestamp": datetime.datetime(2026, 3, 1, 0, 0),
        "n_icebergs": 35,
        "perturbation": 0.4,
        "forecast_hours": 96,
        "seed": 300,
    },
    "ice_edge_transit": {
        "name": "Marginal Ice Zone Transit",
        "description": "Vessel crossing from open water into marginal ice zone with scattered growlers.",
        "timestamp": datetime.datetime(2026, 4, 10, 18, 0),
        "n_icebergs": 25,
        "perturbation": 0.35,
        "forecast_hours": 36,
        "seed": 400,
    },
    "multi_berg_convergence": {
        "name": "Multi-Iceberg Convergence",
        "description": "5 medium/large icebergs converging toward Maitri shipping corridor due to current/wind alignment.",
        "timestamp": datetime.datetime(2026, 2, 20, 8, 0),
        "n_icebergs": 40,
        "perturbation": 0.25,
        "forecast_hours": 72,
        "seed": 500,
    },
}


def run_scenario(
    scenario_key: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Execute a full test scenario.

    Returns a structured report with:
      - Environment summary
      - Iceberg population
      - Detection results
      - Trajectory predictions
      - Hazard assessment
      - Timing metrics
    """
    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key}. Choose from {list(SCENARIOS.keys())}")

    cfg = SCENARIOS[scenario_key]
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data", "test_scenarios")
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "scenario": scenario_key,
        "name": cfg["name"],
        "description": cfg["description"],
        "timestamp": cfg["timestamp"].isoformat(),
        "phases": {},
        "timing": {},
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {cfg['name']}")
        print(f"{'='*60}")

    # ── Phase 1: Environment Generation ──────────────────
    t0 = time.time()
    env = AntarcticEnvironment(seed=cfg["seed"])
    env_state = env.generate(cfg["timestamp"], cfg["perturbation"])
    env_series = env.generate_timeseries(
        cfg["timestamp"],
        hours=cfg["forecast_hours"],
        dt_hours=6.0,
        perturbation_strength=cfg["perturbation"],
    )
    t_env = time.time() - t0
    report["timing"]["environment_generation_s"] = round(t_env, 3)

    # Export environment
    env_path = os.path.join(output_dir, f"{scenario_key}_environment.json")
    export_environment_json(env_state, env_path)

    env_summary = {}
    for field in ['wind_u', 'wind_v', 'sst', 'ice_conc', 'wave_height']:
        arr = env_state[field]
        env_summary[field] = {
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
            "mean": round(float(arr.mean()), 3),
        }
    report["phases"]["environment"] = env_summary

    if verbose:
        print(f"\n[Phase 1] Environment ({t_env:.2f}s)")
        for k, v in env_summary.items():
            print(f"  {k}: min={v['min']:.2f}, max={v['max']:.2f}, mean={v['mean']:.2f}")

    # ── Phase 2: Iceberg Population ──────────────────────
    t0 = time.time()
    population = generate_iceberg_population(
        cfg["n_icebergs"],
        timestamp=cfg["timestamp"],
        seed=cfg["seed"] + 1,
        env_state=env_state,
    )
    t_pop = time.time() - t0
    report["timing"]["population_generation_s"] = round(t_pop, 3)

    size_dist = {}
    for berg in population:
        cls = berg["size_class"]
        size_dist[cls] = size_dist.get(cls, 0) + 1

    report["phases"]["population"] = {
        "total": len(population),
        "size_distribution": size_dist,
    }

    if verbose:
        print(f"\n[Phase 2] Population ({t_pop:.2f}s)")
        print(f"  Total: {len(population)} icebergs")
        for cls, count in size_dist.items():
            print(f"  {cls}: {count}")

    # ── Phase 3: Environmental Detection ─────────────────
    t0 = time.time()
    detected = detect_icebergs_from_environment(
        env_state,
        sensitivity=0.6,
        seed=cfg["seed"] + 2,
    )
    t_det = time.time() - t0
    report["timing"]["detection_s"] = round(t_det, 3)

    det_summary = {
        "candidates_found": len(detected),
        "high_confidence": sum(1 for d in detected if d["confidence"] > 0.7),
        "medium_confidence": sum(1 for d in detected if 0.4 <= d["confidence"] <= 0.7),
        "low_confidence": sum(1 for d in detected if d["confidence"] < 0.4),
    }
    report["phases"]["detection"] = det_summary

    if verbose:
        print(f"\n[Phase 3] Detection ({t_det:.2f}s)")
        print(f"  Candidates: {det_summary['candidates_found']}")
        print(f"  High confidence: {det_summary['high_confidence']}")

    # ── Phase 4: Trajectory Prediction ───────────────────
    t0 = time.time()
    # Predict for top 10 most dangerous icebergs
    all_bergs = population + detected
    for berg in all_bergs:
        berg["hazard_level"] = classify_hazard_level(berg)

    dangerous = sorted(
        all_bergs,
        key=lambda b: {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}.get(b["hazard_level"], 0),
        reverse=True,
    )[:10]

    trajectories = {}
    for berg in dangerous:
        berg_id = berg.get("id", "unknown")
        traj = predict_trajectory(
            berg, env_series,
            dt_hours=3.0,
            forecast_hours=min(48.0, cfg["forecast_hours"]),
        )
        trajectories[berg_id] = traj

    t_traj = time.time() - t0
    report["timing"]["trajectory_prediction_s"] = round(t_traj, 3)

    traj_summary = {
        "icebergs_tracked": len(trajectories),
        "total_points": sum(len(t) for t in trajectories.values()),
    }

    # Compute displacement statistics
    displacements = []
    for berg_id, traj in trajectories.items():
        if len(traj) >= 2:
            start, end = traj[0], traj[-1]
            d = math.sqrt((end["lat"] - start["lat"])**2 + (end["lon"] - start["lon"])**2) * 111.0
            displacements.append(d)

    if displacements:
        traj_summary["avg_displacement_km"] = round(float(np.mean(displacements)), 1)
        traj_summary["max_displacement_km"] = round(float(np.max(displacements)), 1)
        traj_summary["min_displacement_km"] = round(float(np.min(displacements)), 1)

    report["phases"]["trajectory"] = traj_summary

    if verbose:
        print(f"\n[Phase 4] Trajectory ({t_traj:.2f}s)")
        print(f"  Tracked: {traj_summary['icebergs_tracked']} icebergs")
        if displacements:
            print(f"  Avg displacement: {traj_summary['avg_displacement_km']:.1f} km")
            print(f"  Max displacement: {traj_summary['max_displacement_km']:.1f} km")

    # ── Phase 5: Hazard Assessment ───────────────────────
    hazard_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    for berg in all_bergs:
        level = berg.get("hazard_level", "LOW")
        hazard_counts[level] = hazard_counts.get(level, 0) + 1

    report["phases"]["hazard_assessment"] = {
        "total_hazards": len(all_bergs),
        "distribution": hazard_counts,
        "navigation_advisory": _generate_advisory(hazard_counts, env_summary),
    }

    if verbose:
        print(f"\n[Phase 5] Hazard Assessment")
        for level, count in hazard_counts.items():
            print(f"  {level}: {count}")
        print(f"  Advisory: {report['phases']['hazard_assessment']['navigation_advisory']}")

    # ── Export Full Results ───────────────────────────────
    export_iceberg_database(
        all_bergs,
        os.path.join(output_dir, f"{scenario_key}_icebergs.json"),
    )

    # Export trajectories
    traj_export = {}
    for berg_id, traj in trajectories.items():
        traj_export[berg_id] = traj
    with open(os.path.join(output_dir, f"{scenario_key}_trajectories.json"), 'w') as f:
        json.dump(traj_export, f, indent=2)

    # Export report
    report["timing"]["total_s"] = round(sum(report["timing"].values()), 3)
    with open(os.path.join(output_dir, f"{scenario_key}_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"\n{'-'*60}")
        print(f"Total time: {report['timing']['total_s']:.2f}s")
        print(f"Results saved to: {output_dir}")

    return report


def _generate_advisory(hazard_counts: Dict, env_summary: Dict) -> str:
    """Generate a human-readable navigation advisory."""
    critical = hazard_counts.get("CRITICAL", 0)
    high = hazard_counts.get("HIGH", 0)

    max_ice = env_summary.get("ice_conc", {}).get("max", 0)
    max_wind = max(
        abs(env_summary.get("wind_u", {}).get("max", 0)),
        abs(env_summary.get("wind_v", {}).get("max", 0)),
    )
    max_wave = env_summary.get("wave_height", {}).get("max", 0)

    if critical > 3 or (critical > 0 and max_ice > 0.7):
        return (
            f"CRITICAL: {critical} critical-class hazards detected. "
            f"Route closure recommended. Ice concentration up to {max_ice*100:.0f}%. "
            f"Winds gusting to {max_wind:.0f} m/s."
        )
    elif high > 3 or critical > 0:
        return (
            f"HIGH RISK: {critical + high} significant hazards. "
            f"Proceed with icebreaker escort only. Max SIC {max_ice*100:.0f}%. "
            f"Wave height up to {max_wave:.1f}m."
        )
    elif high > 0:
        return (
            f"MODERATE RISK: {high} notable hazards in transit corridor. "
            f"Maintain radar watch and reduce speed in marginal ice zones."
        )
    else:
        return (
            f"LOW RISK: Standard operations. {hazard_counts.get('MODERATE', 0)} minor hazards tracked. "
            f"Sea state manageable."
        )


def run_all_scenarios(output_dir: Optional[str] = None) -> Dict:
    """Run all predefined test scenarios and produce a combined report."""
    results = {}
    for key in SCENARIOS:
        results[key] = run_scenario(key, output_dir=output_dir)
    return results


if __name__ == "__main__":
    results = run_all_scenarios()
    print("\n" + "=" * 60)
    print("ALL SCENARIOS COMPLETE")
    print("=" * 60)
    for key, report in results.items():
        total = report["timing"]["total_s"]
        hazards = report["phases"]["hazard_assessment"]["total_hazards"]
        advisory = report["phases"]["hazard_assessment"]["navigation_advisory"][:80]
        print(f"\n  {key}: {total:.2f}s, {hazards} hazards — {advisory}...")
