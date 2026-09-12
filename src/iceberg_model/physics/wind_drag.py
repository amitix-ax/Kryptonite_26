"""
Wind (air) drag on the above-water iceberg sail.

    F_air = 0.5 * rho_air * C_Da * A_air * |U_air - U_i| * (U_air - U_i)
    a_air = F_air / m

Deliberately NOT using the common "iceberg drifts at ~2% of wind speed"
heuristic as the primary model (spec section 13) — that rule is a derived
rule-of-thumb from a force balance, not a substitute for one. We compute the
force explicitly instead; the 2% behavior should emerge naturally from the
force balance in forces.py/dynamics.py rather than being hardcoded.

Sail area A_air uses the freeboard (above-water thickness) times width, a
simplification consistent with the rectangular-block geometry used
elsewhere (see docs/physics.md).
"""

from __future__ import annotations

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.buoyancy import calculate_freeboard
from iceberg_model.physics.constants import EPSILON
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def sail_area(state: IcebergState, config: PhysicsConfig) -> float:
    """Above-water cross-sectional area facing the wind [m^2]."""
    freeboard = calculate_freeboard(state, config)
    return state.W * max(freeboard, 0.0)


def calculate_wind_drag_acceleration(state: IcebergState, env: EnvironmentalState,
                                      config: PhysicsConfig, mass: float) -> np.ndarray:
    U_air = env.wind_velocity
    if U_air is None or not env.validity_mask.get("wind_velocity", True):
        return np.zeros(2)

    U_i = state.velocity
    rel = U_air - U_i
    speed = np.linalg.norm(rel)
    A_air = sail_area(state, config)
    rho_air = config.fluids.rho_air
    C_Da = config.drag.c_da

    F = 0.5 * rho_air * C_Da * A_air * speed * rel
    return F / max(mass, EPSILON)
