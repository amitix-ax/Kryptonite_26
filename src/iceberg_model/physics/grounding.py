"""
Grounding: EVENT-driven, not a continuously-blended force (spec section 17).

Trigger condition, evaluated by numerical/events.py as a solve_ivp event:

    D >= H_water(x, y)

where D is keel depth (physics/buoyancy.py::calculate_submerged_depth) and
H_water is local bathymetry (positive-down water depth).

Once grounded, free-floating dynamics do NOT continue unchanged: the
dynamics mode switches to GROUNDED (see numerical/deterministic_solver.py),
and horizontal acceleration is dominated by Coulomb-style friction opposing
motion:

    F_friction = -mu * N * v / (|v| + epsilon)

Normal force N (prototype model, `normal_force_model="weight_minus_buoyancy"`
in PhysicsConfig): the portion of iceberg weight not supported by buoyancy,
i.e. the excess weight pressing into the seabed:

    N = max(0, (m - rho_w * V_sub) * g)

This is a deliberately simple PROTOTYPE model. It does NOT include sediment
resistance, seabed slope effects, grounding geometry, or partial grounding —
those are called out as future extensions in docs/physics.md and the
project spec (section 17).
"""

from __future__ import annotations

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.buoyancy import calculate_submerged_depth
from iceberg_model.physics.constants import EPSILON, GRAVITY
from iceberg_model.state.iceberg_state import IcebergState


def is_grounded(state: IcebergState, config: PhysicsConfig, water_depth_m: float) -> bool:
    """Grounding condition: D >= H_water(x, y)."""
    if water_depth_m is None:
        return False
    D = calculate_submerged_depth(state, config)
    return D >= water_depth_m


def normal_force(state: IcebergState, config: PhysicsConfig, mass: float) -> float:
    """N = max(0, (m - rho_w * V_sub) * g). See module docstring."""
    rho_w = config.fluids.rho_seawater
    submerged_fraction = config.fluids.rho_ice / rho_w
    V_sub = submerged_fraction * state.volume
    N = (mass - rho_w * V_sub) * GRAVITY
    return max(N, 0.0)


def calculate_grounding_acceleration(state: IcebergState, config: PhysicsConfig,
                                      mass: float, grounded: bool) -> np.ndarray:
    if not grounded or not config.grounding.enabled:
        return np.zeros(2)

    mu = config.grounding.friction_coefficient_mu
    N = normal_force(state, config, mass)
    v = state.velocity
    speed = np.linalg.norm(v)

    F_friction = -mu * N * v / (speed + EPSILON)
    return F_friction / max(mass, EPSILON)
