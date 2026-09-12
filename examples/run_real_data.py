"""
Example: run a full simulation end-to-end and export the trajectory.

This uses the synthetic in-memory environment (examples/synthetic_environment.py)
so it runs with zero external data files -- swap `build_synthetic_environment()`
for a real GeoTIFF-backed EnvironmentalDataset (built via
data/geotiff_reader.py + data/reprojection.py) to run on real data.

Run with:
    python examples/run_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_environment import build_synthetic_environment

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.io.trajectory_export import export_csv
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.simulation.simulator import Simulator
from iceberg_model.state.iceberg_state import IcebergState
from iceberg_model.validation.physical_checks import check_state


def main():
    config = PhysicsConfig()
    environment = build_synthetic_environment()

    simulator = Simulator(config=config, environment=environment, melt_rates_config=MeltRatesConfig())

    initial_state = IcebergState(x=0.0, y=0.0, u=0.0, v=0.0, L=800.0, W=400.0, H=150.0)
    t_end = 24 * 3600.0  # 24 hours

    result = simulator.run(initial_state, t_end=t_end)

    times, states = result.concatenated()
    final_state = IcebergState.from_vector(states[-1], time=times[-1], mode=result.final_mode)

    report = check_state(final_state, config, water_depth_m=500.0)
    print(f"Simulated {len(times)} steps over {t_end/3600:.1f} hours.")
    print(f"Final position: x={final_state.x:.1f} m, y={final_state.y:.1f} m")
    print(f"Final mode: {result.final_mode.value}")
    print(f"Events logged: {result.event_log.events}")
    print(f"Physical check: ok={report.ok}, violations={report.violations}")

    out_path = Path(__file__).resolve().parents[1] / "data" / "example_trajectory.csv"
    export_csv(result, out_path)
    print(f"Trajectory written to: {out_path}")


if __name__ == "__main__":
    main()
