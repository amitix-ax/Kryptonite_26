"""
Central force/acceleration aggregation (spec section 19).

    a_total = a_ocean + a_air + a_coriolis + a_pressure + a_seaice + a_grounding

Returns a structured ForceBreakdown so every component is individually
inspectable -- critical for debugging and for later ConvLSTM residual
analysis (the residual model compares v_observed against v_physics, and
having the per-force breakdown makes it possible to diagnose *which*
physical term the correction is compensating for).
"""

from __future__ import annotations

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.constants import RHO_ICE_DEFAULT
from iceberg_model.physics.coriolis import calculate_coriolis_acceleration
from iceberg_model.physics.grounding import calculate_grounding_acceleration, is_grounded
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.physics.ocean_drag import calculate_ocean_drag_acceleration
from iceberg_model.physics.pressure_gradient import calculate_pressure_gradient_acceleration
from iceberg_model.physics.sea_ice_drag import calculate_sea_ice_drag_acceleration
from iceberg_model.physics.wind_drag import calculate_wind_drag_acceleration
from iceberg_model.state.iceberg_state import (
    EnvironmentalState,
    ForceBreakdown,
    IcebergState,
)


def iceberg_mass(state: IcebergState, config: PhysicsConfig) -> float:
    """m = rho_i * V, V = L*W*H (spec section 5). Document the geometric
    simplification: a rectangular-block approximation. Replace with
    ellipsoid/polygon/observed geometry by swapping IcebergState.volume."""
    return config.fluids.rho_ice * state.volume


def calculate_total_acceleration(state: IcebergState, env: EnvironmentalState,
                                  config: PhysicsConfig,
                                  grounded_override: bool | None = None) -> ForceBreakdown:
    """
    Compute the full ForceBreakdown for the current state/environment.

    `grounded_override`: if provided, bypasses the grounding-condition
    recheck (used by the solver once an event has already determined the
    mode for this step, to avoid re-deriving it from possibly-stale
    bathymetry sampling mid-step).
    """
    mass = iceberg_mass(state, config)

    if grounded_override is not None:
        grounded = grounded_override
    else:
        grounded = (
            config.grounding.enabled
            and env.bathymetry is not None
            and env.validity_mask.get("bathymetry", True)
            and is_grounded(state, config, env.bathymetry)
        )

    if grounded:
        # Per spec section 17: grounded dynamics do NOT reuse free-floating
        # forces unchanged. Ocean/air/sea-ice/pressure/coriolis forces are
        # suppressed in favor of the friction-dominated grounded regime.
        a_ocean = _zeros()
        a_air = _zeros()
        a_coriolis = _zeros()
        a_pressure = _zeros()
        a_sea_ice = _zeros()
        a_grounding = calculate_grounding_acceleration(state, config, mass, grounded=True)
    else:
        a_ocean = calculate_ocean_drag_acceleration(state, env, config, mass).acceleration
        a_air = calculate_wind_drag_acceleration(state, env, config, mass)
        a_coriolis = calculate_coriolis_acceleration(state, env, enabled=config.coriolis_enabled)
        a_pressure = calculate_pressure_gradient_acceleration(state, env, config)
        a_sea_ice = calculate_sea_ice_drag_acceleration(state, env, config, mass)
        a_grounding = _zeros()

    breakdown = ForceBreakdown(
        ocean=a_ocean,
        air=a_air,
        coriolis=a_coriolis,
        pressure=a_pressure,
        sea_ice=a_sea_ice,
        grounding=a_grounding,
        total=_zeros(),
    )
    breakdown.recompute_total()
    return breakdown


def _zeros():
    import numpy as np
    return np.zeros(2)
