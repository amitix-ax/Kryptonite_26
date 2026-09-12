"""
Optional pressure-gradient force.

Full form:

    F_pressure = - INTEGRAL_Vsub grad_h(p) dV
    a_pressure = F_pressure / m

Data reality: full 3D pressure fields are rarely available. When only
sea-surface height (SSH) is available, we use the standard simplified
geostrophic-style parameterization for the horizontal pressure gradient
force per unit mass, assuming a barotropic (depth-independent) pressure
gradient set by the sea-surface slope:

    a_pressure = -g * grad_h(SSH)

This is a simplification: it ignores baroclinic (density-driven) structure
below the surface, which would require the full density/pressure field the
spec explicitly says we should not fabricate. If ssh_gradient is
unavailable, or config.pressure_gradient.enabled is False, this term
returns zero — it is NEVER estimated from a missing field.
"""

from __future__ import annotations

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.constants import GRAVITY
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def calculate_pressure_gradient_acceleration(state: IcebergState, env: EnvironmentalState,
                                              config: PhysicsConfig) -> np.ndarray:
    if not config.pressure_gradient.enabled:
        return np.zeros(2)
    if env.ssh_gradient is None or not env.validity_mask.get("ssh_gradient", True):
        return np.zeros(2)

    grad = np.asarray(env.ssh_gradient, dtype=float)
    return -GRAVITY * grad
