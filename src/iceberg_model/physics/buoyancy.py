"""
Buoyancy via Archimedes' principle.

    rho_i * V * g = rho_w * V_sub * g
    =>  V_sub / V = rho_i / rho_w

For a rectangular-block iceberg approximation (V = L*W*H, uniform cross
section), the submerged fraction of volume equals the submerged fraction
of thickness, so:

    D ~= (rho_i / rho_w) * H          (approximate keel depth)
    freeboard = H - D
"""

from __future__ import annotations

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.state.iceberg_state import IcebergState


def calculate_submerged_depth(state: IcebergState, config: PhysicsConfig) -> float:
    """Return approximate keel depth D [m] for a rectangular-block iceberg.

    Validity: 0 < D < H is expected for a freely floating, physically
    sane iceberg. Callers are responsible for comparing D against local
    bathymetry to decide whether grounding should trigger.
    """
    rho_i = config.fluids.rho_ice
    rho_w = config.fluids.rho_seawater
    if not (0 < rho_i < rho_w):
        raise ValueError(
            f"Invalid density configuration: rho_ice={rho_i}, rho_seawater={rho_w}. "
            "Require 0 < rho_ice < rho_seawater."
        )
    D = (rho_i / rho_w) * state.H
    return D


def calculate_freeboard(state: IcebergState, config: PhysicsConfig) -> float:
    """Return freeboard (above-water thickness) [m]. freeboard = H - D."""
    D = calculate_submerged_depth(state, config)
    return state.H - D


def submerged_fraction(config: PhysicsConfig) -> float:
    """Return rho_i / rho_w, the fraction of iceberg volume submerged."""
    return config.fluids.rho_ice / config.fluids.rho_seawater


def submerged_cross_sectional_area(state: IcebergState, config: PhysicsConfig) -> float:
    """
    Effective submerged drag area A_k [m^2] used by ocean_drag.py in
    'depth_averaged' mode: a simple rectangular keel cross-section facing
    the relative current, approximated as width * keel_depth.

    This is a coarse simplification (see docs/physics.md); depth-integrated
    mode (ocean_drag.py) replaces this with an explicit vertical integral
    when a current profile is available.
    """
    D = calculate_submerged_depth(state, config)
    return state.W * D
