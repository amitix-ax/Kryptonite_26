"""
Typed representations of the iceberg physical state, sampled environment,
and force/acceleration breakdown.

The state vector (see docs/physics.md) is:

    s = [x, y, u, v, L, W, H]

x, y   - projected east/north position [m], in PhysicsConfig.crs.working_crs
u, v   - eastward/northward iceberg velocity [m/s]
L, W, H - iceberg length, width, thickness [m]

We deliberately do NOT integrate latitude/longitude directly (principle in
section 4 of the spec) — lat/lon conversion happens only at I/O boundaries
via data/reprojection.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class DynamicsMode(str, Enum):
    FREE_FLOATING = "FREE_FLOATING"
    GROUNDED = "GROUNDED"
    MELTING_ONLY = "MELTING_ONLY"
    TERMINATED = "TERMINATED"


@dataclass
class IcebergState:
    """The 7-element physical state vector plus bookkeeping fields."""

    x: float          # [m], projected easting
    y: float          # [m], projected northing
    u: float          # [m/s], eastward velocity
    v: float          # [m/s], northward velocity
    L: float          # [m], length
    W: float          # [m], width
    H: float          # [m], thickness
    time: float = 0.0  # [s], simulation time (seconds since epoch/start)
    mode: DynamicsMode = DynamicsMode.FREE_FLOATING

    def as_vector(self) -> np.ndarray:
        """Return the 7-element ODE state vector [x,y,u,v,L,W,H]."""
        return np.array([self.x, self.y, self.u, self.v, self.L, self.W, self.H], dtype=float)

    @classmethod
    def from_vector(cls, vec: np.ndarray, time: float = 0.0,
                     mode: DynamicsMode = DynamicsMode.FREE_FLOATING) -> "IcebergState":
        x, y, u, v, L, W, H = vec
        return cls(x=x, y=y, u=u, v=v, L=L, W=W, H=H, time=time, mode=mode)

    @property
    def volume(self) -> float:
        """V = L * W * H. See docs/physics.md for the geometric simplification."""
        return self.L * self.W * self.H

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.u, self.v])

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def is_geometrically_valid(self, min_dimension_m: float) -> bool:
        return self.L > min_dimension_m and self.W > min_dimension_m and self.H > min_dimension_m


@dataclass
class EnvironmentalState:
    """
    Environmental fields sampled at a given (x, y, t), returned by
    data/environmental_dataset.py::sample_environment().

    Any field that could not be resolved (missing data, out-of-bounds,
    masked/nodata) MUST be None, with the corresponding key in
    `validity_mask` set to False. Consumers (physics/*) must treat a None
    field as "force disabled for this step", never silently substitute a
    fabricated value. See engineering principle J.
    """

    ocean_velocity: Optional[np.ndarray] = None      # [u_o, v_o] m/s
    wind_velocity: Optional[np.ndarray] = None        # [u_air, v_air] m/s
    sea_ice_concentration: Optional[float] = None      # [0-1]
    sea_ice_velocity: Optional[np.ndarray] = None     # [u_si, v_si] m/s (may be a fallback)
    sst: Optional[float] = None                        # sea surface temperature [C]
    ssh: Optional[float] = None                         # sea surface height [m]
    ssh_gradient: Optional[np.ndarray] = None           # [dssh/dx, dssh/dy]
    bathymetry: Optional[float] = None                   # water depth, positive down [m]
    latitude_deg: Optional[float] = None                 # needed for Coriolis
    ocean_velocity_profile: Optional[np.ndarray] = None  # shape (n_z, 2), depth-integrated mode
    depth_levels_m: Optional[np.ndarray] = None           # shape (n_z,), matches profile above

    validity_mask: dict = field(default_factory=dict)
    """Maps field name -> bool. False/missing means the field was not
    available at this (x, y, t) and any dependent force must be skipped
    (not fabricated)."""

    sea_ice_velocity_is_fallback: bool = False
    """True if sea_ice_velocity was substituted from a fallback source
    (e.g. ocean current) rather than an actual sea-ice velocity product."""


@dataclass
class ForceBreakdown:
    """
    Structured decomposition of the total acceleration, one 2-vector [m/s^2]
    per physical mechanism. Central to explainability/debugging per
    engineering requirement in section 19 of the spec.
    """

    ocean: np.ndarray
    air: np.ndarray
    coriolis: np.ndarray
    pressure: np.ndarray
    sea_ice: np.ndarray
    grounding: np.ndarray
    total: np.ndarray

    @classmethod
    def zeros(cls) -> "ForceBreakdown":
        z = lambda: np.zeros(2)
        return cls(ocean=z(), air=z(), coriolis=z(), pressure=z(), sea_ice=z(),
                    grounding=z(), total=z())

    def recompute_total(self) -> None:
        self.total = self.ocean + self.air + self.coriolis + self.pressure + self.sea_ice + self.grounding
