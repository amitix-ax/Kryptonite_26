"""
Empirical sea-ice interaction drag (spec section 16).

    F_si = 0.5 * rho_eff * C_Dsi * A_si * C_ice * |U_si - U_i| * (U_si - U_i)
    a_si = F_si / m

This is explicitly an empirical parameterization representing the
aggregate effect of sea-ice floes pushing/dragging on the iceberg,
scaled by local sea-ice concentration C_ice in [0, 1]. It is NOT a
first-principles force and has no universally agreed coefficient —
C_Dsi must be calibrated.

rho_eff defaults to seawater density (the sea-ice field is treated as a
momentum-coupled fluid-like medium at the surface); this is a modeling
choice documented here, not a literature constant.

U_si (sea-ice velocity) is frequently unavailable. If
EnvironmentalState.sea_ice_velocity is None, this module does NOT
fabricate a value — resolution of a fallback (e.g. substituting ocean
current) happens upstream in environmental_dataset.py and is flagged via
`sea_ice_velocity_is_fallback`. If neither is present the term is zero.
"""

from __future__ import annotations

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.constants import EPSILON
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def sea_ice_contact_area(state: IcebergState) -> float:
    """Approximate lateral contact area between iceberg and surrounding
    sea-ice floes: width * a nominal 1 m interaction band at the waterline.
    This is a coarse placeholder — see docs/physics.md limitations."""
    return state.W * 1.0


def calculate_sea_ice_drag_acceleration(state: IcebergState, env: EnvironmentalState,
                                         config: PhysicsConfig, mass: float) -> np.ndarray:
    if not config.sea_ice_drag.enabled:
        return np.zeros(2)

    C_ice = env.sea_ice_concentration
    U_si = env.sea_ice_velocity
    if C_ice is None or U_si is None:
        return np.zeros(2)
    if not env.validity_mask.get("sea_ice_concentration", True):
        return np.zeros(2)

    C_ice = float(np.clip(C_ice, 0.0, 1.0))
    if C_ice <= 0.0:
        return np.zeros(2)

    U_i = state.velocity
    rel = U_si - U_i
    speed = np.linalg.norm(rel)
    A_si = sea_ice_contact_area(state)
    rho_eff = config.fluids.rho_seawater
    C_Dsi = config.drag.c_dsi

    F = 0.5 * rho_eff * C_Dsi * A_si * C_ice * speed * rel
    return F / max(mass, EPSILON)
