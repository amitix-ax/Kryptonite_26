"""
Physical sanity checks -- run against every simulated state to catch
non-physical output before it is trusted downstream. These are assertions
about the physics, not statistical accuracy metrics (see metrics.py /
trajectory_metrics.py for those).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.state.iceberg_state import IcebergState


@dataclass
class PhysicalCheckReport:
    ok: bool = True
    violations: list = field(default_factory=list)

    def add(self, message: str) -> None:
        self.ok = False
        self.violations.append(message)


def check_state(state: IcebergState, config: PhysicsConfig,
                 water_depth_m: float | None = None) -> PhysicalCheckReport:
    report = PhysicalCheckReport()

    if state.L <= 0 or state.W <= 0 or state.H <= 0:
        report.add(f"Non-positive geometry: L={state.L}, W={state.W}, H={state.H}")

    rho_i, rho_w = config.fluids.rho_ice, config.fluids.rho_seawater
    if not (0 < rho_i < rho_w):
        report.add(f"Density constraint violated: rho_ice={rho_i}, rho_seawater={rho_w}")

    from iceberg_model.physics.buoyancy import calculate_submerged_depth
    D = calculate_submerged_depth(state, config)
    if water_depth_m is not None and not (0 < D < water_depth_m) and state.mode.value == "FREE_FLOATING":
        report.add(f"Submerged depth D={D} not within (0, water_depth={water_depth_m}) while free-floating")

    speed = np.linalg.norm(state.velocity)
    if speed > 5.0:
        # Icebergs do not physically drift at multiple m/s under
        # realistic wind/current forcing; flag as suspicious, not fatal.
        report.add(f"Suspiciously high drift speed: {speed:.3f} m/s (sanity threshold 5 m/s)")

    if not np.all(np.isfinite(state.as_vector())):
        report.add("Non-finite value in state vector (NaN/Inf).")

    return report
