"""
Equations of motion (spec section 20) -- the ODE right-hand-side.

    dx/dt = u
    dy/dt = v
    du/dt = a_x(state, environment)
    dv/dt = a_y(state, environment)
    dL/dt = melt_length_rate
    dW/dt = melt_width_rate
    dH/dt = melt_thickness_rate

This is the ONE place a numerical integrator calls into. It is pure
physics: no ML, no residual correction. The physics engine must run
end-to-end with this function alone (ResidualModel = None is a fully
supported configuration -- see uncertainty/residuals.py).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.forces import calculate_total_acceleration
from iceberg_model.physics.grounding import is_grounded
from iceberg_model.physics.melting import MeltRatesConfig, calculate_melt_rates
from iceberg_model.state.iceberg_state import DynamicsMode, EnvironmentalState, ForceBreakdown, IcebergState

# An EnvironmentalProvider maps (x, y, t) -> EnvironmentalState.
EnvironmentalProvider = Callable[[float, float, float], EnvironmentalState]


def dynamics(t: float, state_vec: np.ndarray, environmental_provider: EnvironmentalProvider,
             config: PhysicsConfig, melt_rates_config: MeltRatesConfig,
             mode: DynamicsMode = DynamicsMode.FREE_FLOATING,
             last_breakdown: Optional[list] = None) -> np.ndarray:
    """
    ODE right-hand-side compatible with scipy.integrate.solve_ivp's
    `fun(t, y)` signature via a partial/closure (see numerical/deterministic_solver.py).

    `last_breakdown`, if provided as a single-element list, is mutated in
    place to stash the most recent ForceBreakdown for diagnostics/logging
    (solve_ivp does not let the RHS return auxiliary data directly).
    """
    state = IcebergState.from_vector(state_vec, time=t, mode=mode)
    env = environmental_provider(state.x, state.y, t)

    if mode == DynamicsMode.TERMINATED:
        return np.zeros(7)

    if mode == DynamicsMode.MELTING_ONLY:
        melt = calculate_melt_rates(state, env, config, melt_rates_config)
        return np.array([0.0, 0.0, 0.0, 0.0, melt.dL_dt, melt.dW_dt, melt.dH_dt])

    grounded = (mode == DynamicsMode.GROUNDED)
    breakdown: ForceBreakdown = calculate_total_acceleration(state, env, config, grounded_override=grounded)
    if last_breakdown is not None:
        last_breakdown.clear()
        last_breakdown.append(breakdown)

    ax, ay = breakdown.total

    melt = calculate_melt_rates(state, env, config, melt_rates_config) if config.melting.enabled else None
    dL_dt = melt.dL_dt if melt else 0.0
    dW_dt = melt.dW_dt if melt else 0.0
    dH_dt = melt.dH_dt if melt else 0.0

    return np.array([state.u, state.v, ax, ay, dL_dt, dW_dt, dH_dt])
