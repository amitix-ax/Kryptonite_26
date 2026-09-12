"""
Physical constants used throughout the iceberg drift model.

IMPORTANT: Every constant here is a DEFAULT. None of these values should be
treated as ground truth for a specific iceberg or region — they are typical
literature values intended to make the model runnable out of the box. All
of them are re-exposed as configurable fields in
`iceberg_model.config.physics_config.PhysicsConfig`, and any call site that
needs a "real" value should pull it from a PhysicsConfig instance, not from
this module directly.

Sources (typical ranges cited in sea-ice / iceberg drift literature):
- Earth's rotation rate, seawater/air densities, gravity: standard
  geophysical reference values (e.g. Gill, "Atmosphere-Ocean Dynamics", 1982).
- Iceberg drag coefficients (C_Dw, C_Da) and iceberg density: commonly cited
  ranges from Bigg et al. (1997), Smith (1993), and related Antarctic
  iceberg drift modeling papers. These vary substantially by iceberg
  geometry/roughness and MUST be tuned/validated against observations for
  any real deployment — they are not universal constants the way g or
  Omega are.

Do not add new "magic numbers" to this file without a citation in a
docstring/comment. If a value cannot be sourced, it must live in a config
default with an explicit "UNVALIDATED DEFAULT" comment instead.
"""

from __future__ import annotations

# --- Universal / geophysical constants -------------------------------------

GRAVITY: float = 9.80665
"""Standard gravity [m/s^2]."""

EARTH_ROTATION_RATE: float = 7.2921159e-5
"""Earth's angular rotation rate, Omega [rad/s]."""

# --- Reference fluid properties (defaults — override via PhysicsConfig) ----

RHO_SEAWATER_DEFAULT: float = 1027.0
"""Reference seawater density [kg/m^3]. Typical polar surface value; varies
with temperature/salinity — treat as a tunable default, not a universal
constant."""

RHO_AIR_DEFAULT: float = 1.225
"""Reference air density at sea level, ~15C, standard atmosphere [kg/m^3]."""

RHO_ICE_DEFAULT: float = 900.0
"""Reference glacial (iceberg) ice density [kg/m^3]. Literature range is
approximately 850-920 kg/m^3 depending on air-bubble content; 900 is a
commonly used mid-range default (e.g. Bigg et al., 1997) and MUST be
validated/tuned for the specific iceberg population being modeled."""

# --- Drag coefficients (UNVALIDATED DEFAULTS — must be tuned) --------------

C_DW_DEFAULT: float = 1.5
"""Ocean (water) drag coefficient, dimensionless. Commonly cited range in
iceberg drift literature is roughly 0.9-2.5 depending on keel roughness and
shape assumptions. This default is a starting point, not a validated value."""

C_DA_DEFAULT: float = 1.3
"""Air (wind) drag coefficient, dimensionless. Commonly cited range is
roughly 1.0-2.0 for above-water iceberg sail geometry."""

C_DSI_DEFAULT: float = 1.0
"""Sea-ice interaction drag coefficient, dimensionless. This is an
empirical parameterization coefficient (see physics/sea_ice_drag.py) and
has no single agreed-upon literature value — it must be calibrated."""

# --- Numerical safety -------------------------------------------------------

EPSILON: float = 1e-9
"""Small number used to avoid division-by-zero in normalization of near-zero
vectors (e.g. v / (|v| + EPSILON))."""
