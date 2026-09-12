"""
Coriolis acceleration.

    f = 2 * Omega * sin(phi)

Coordinate convention (explicit, per spec section 14):
    - x is EASTING, y is NORTHING (standard projected/local-tangent-plane
      convention used throughout this codebase).
    - u = dx/dt (eastward component), v = dy/dt (northward component).
    - a_coriolis = [ f*v, -f*u ]   for the Southern Hemisphere sign
      convention used here (phi is negative south of the equator, so f is
      negative in the Antarctic — this naturally deflects moving objects to
      the LEFT of their velocity, which is the correct Southern-Hemisphere
      behavior).

This sign convention is verified in tests/test_coriolis.py: for a
Southern-Hemisphere latitude (phi < 0) and a purely eastward velocity
(u>0, v=0), the resulting acceleration must have a negative y-component
(deflection to the left / equatorward-then-left when viewed along the
direction of motion), consistent with a negative f.
"""

from __future__ import annotations

import numpy as np

from iceberg_model.physics.constants import EARTH_ROTATION_RATE
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def coriolis_parameter(latitude_deg: float) -> float:
    """f = 2 * Omega * sin(phi), phi in radians. Negative in the Southern Hemisphere."""
    phi = np.radians(latitude_deg)
    return 2.0 * EARTH_ROTATION_RATE * np.sin(phi)


def calculate_coriolis_acceleration(state: IcebergState, env: EnvironmentalState,
                                     enabled: bool = True) -> np.ndarray:
    if not enabled:
        return np.zeros(2)
    if env.latitude_deg is None or not env.validity_mask.get("latitude_deg", True):
        # Missing latitude -> cannot compute a physically meaningful f.
        return np.zeros(2)

    f = coriolis_parameter(env.latitude_deg)
    u, v = state.u, state.v
    return np.array([f * v, -f * u])
