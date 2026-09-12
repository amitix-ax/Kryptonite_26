"""
Ocean (hydrodynamic) drag on the submerged iceberg keel.

Depth-averaged (default, section 11 of spec):

    F_ocean = 0.5 * rho_w * C_Dw * A_k * |U_o - U_i| * (U_o - U_i)
    a_ocean = F_ocean / m

Depth-integrated (section 12, used when a vertical current profile is
supplied via EnvironmentalState.ocean_velocity_profile / depth_levels_m):

    F_ocean = 0.5 * rho_w * C_Dw * INTEGRAL_0^D A(z) |U(z)-U_i| (U(z)-U_i) dz

evaluated via trapezoidal quadrature over the levels that fall within
[0, D] (keel depth). If no profile is available, this function transparently
falls back to depth-averaged and marks the approximation via the returned
`used_depth_averaged_fallback` flag so callers/logs can see it happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig, OceanDragMode
from iceberg_model.physics.buoyancy import calculate_submerged_depth, submerged_cross_sectional_area
from iceberg_model.physics.constants import EPSILON
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


@dataclass
class OceanDragResult:
    acceleration: np.ndarray  # [ax, ay] m/s^2
    mode_used: OceanDragMode
    used_depth_averaged_fallback: bool


def _drag_accel_depth_averaged(state: IcebergState, env: EnvironmentalState,
                                config: PhysicsConfig, mass: float) -> np.ndarray:
    U_o = env.ocean_velocity
    if U_o is None or not env.validity_mask.get("ocean_velocity", True):
        # Principle J: missing data never silently becomes a valid force.
        return np.zeros(2)

    U_i = state.velocity
    rel = U_o - U_i
    speed = np.linalg.norm(rel)
    A_k = submerged_cross_sectional_area(state, config)
    rho_w = config.fluids.rho_seawater
    C_Dw = config.drag.c_dw

    F = 0.5 * rho_w * C_Dw * A_k * speed * rel
    return F / max(mass, EPSILON)


def _drag_accel_depth_integrated(state: IcebergState, env: EnvironmentalState,
                                  config: PhysicsConfig, mass: float) -> tuple[np.ndarray, bool]:
    profile = env.ocean_velocity_profile
    levels = env.depth_levels_m
    if profile is None or levels is None or len(levels) < 2:
        # No profile available -- fall back to depth-averaged, flagged.
        return _drag_accel_depth_averaged(state, env, config, mass), True

    D = calculate_submerged_depth(state, config)
    W = state.W
    U_i = state.velocity
    rho_w = config.fluids.rho_seawater
    C_Dw = config.drag.c_dw

    mask = levels <= D
    if mask.sum() < 2:
        return _drag_accel_depth_averaged(state, env, config, mass), True

    z = levels[mask]
    U_z = profile[mask]  # shape (n, 2)
    rel = U_z - U_i  # broadcast, shape (n, 2)
    speed = np.linalg.norm(rel, axis=1)
    integrand = (W * speed[:, None] * rel)  # A(z) ~= W (unit width strip); shape (n, 2)

    F = 0.5 * rho_w * C_Dw * np.trapz(integrand, z, axis=0)
    return F / max(mass, EPSILON), False


def calculate_ocean_drag_acceleration(state: IcebergState, env: EnvironmentalState,
                                       config: PhysicsConfig, mass: float) -> OceanDragResult:
    """Compute a_ocean [m/s^2] according to config.ocean_drag_mode."""
    if config.ocean_drag_mode == OceanDragMode.DEPTH_INTEGRATED:
        accel, fallback = _drag_accel_depth_integrated(state, env, config, mass)
        return OceanDragResult(accel, OceanDragMode.DEPTH_INTEGRATED, fallback)

    accel = _drag_accel_depth_averaged(state, env, config, mass)
    return OceanDragResult(accel, OceanDragMode.DEPTH_AVERAGED, False)
