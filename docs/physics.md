# Physics Model

This document describes the physical model implemented in
`src/iceberg_model/physics/` and `state/iceberg_state.py`. It is the
authoritative reference for what each force represents, its assumptions,
and its known limitations. Code docstrings mirror this document; if they
ever disagree, treat that as a bug.

## State vector

```
s(t) = [x, y, u, v, L, W, H]
```

- `x, y` — projected easting/northing position [m], in the working CRS
  (default `EPSG:3031`, Antarctic Polar Stereographic; configurable via
  `PhysicsConfig.crs.working_crs`).
- `u, v` — eastward/northward iceberg velocity [m/s].
- `L, W, H` — iceberg length, width, thickness [m].

Latitude/longitude are **never** integrated directly. Conversion to/from
geographic coordinates happens only at I/O boundaries (`io/trajectory_export.py`,
`pyproj`-based).

## Geometry and mass

The iceberg is approximated as a rectangular block:

```
V = L * W * H
m = rho_i * V
```

This is a deliberate simplification (spec section 5). It is the *default*
geometry model; the architecture (`IcebergState.volume`) is written so it
can be swapped for an ellipsoid, an observed polygon footprint, or tabular
iceberg dimensions without touching the force/dynamics code.

## Buoyancy (Archimedes)

```
rho_i * V * g = rho_w * V_sub * g   =>   V_sub / V = rho_i / rho_w
```

For the rectangular-block approximation this reduces to a simple
thickness ratio:

```
D ≈ (rho_i / rho_w) * H        (approximate keel depth)
freeboard = H - D
```

`rho_i < rho_w` is enforced as a hard validation constraint in
`PhysicsConfig` — an invalid density configuration raises at config
construction time, not silently downstream.

## Forces

All forces are computed in `physics/forces.py::calculate_total_acceleration`,
which returns a `ForceBreakdown` (one 2D acceleration vector per
mechanism) — kept structured rather than pre-summed so each term is
individually inspectable for debugging and for future ConvLSTM residual
analysis.

### Ocean drag (`ocean_drag.py`)

```
F_ocean = 0.5 * rho_w * C_Dw * A_k * |U_o - U_i| * (U_o - U_i)
```

Two modes:
- `depth_averaged` (default): `A_k = W * D` (keel cross-section facing
  the current), using a single depth-averaged ocean velocity.
- `depth_integrated`: numerically integrates the relative-velocity drag
  over an explicit vertical current profile (`EnvironmentalState.ocean_velocity_profile`),
  when available. Falls back to depth-averaged (flagged) if no profile
  is supplied.

### Wind drag (`wind_drag.py`)

```
F_air = 0.5 * rho_air * C_Da * A_air * |U_air - U_i| * (U_air - U_i)
```

`A_air = W * freeboard`. We do **not** use the common "iceberg drifts at
~2% of wind speed" heuristic as the model itself — that ratio is an
emergent consequence of the force balance between wind and ocean drag,
not a substitute for computing the forces explicitly.

### Coriolis (`coriolis.py`)

```
f = 2 * Omega * sin(phi)
a_coriolis = [f*v, -f*u]
```

Coordinate convention: x=easting, y=northing, u=dx/dt, v=dy/dt. This
sign convention correctly deflects moving objects to the **left** of
their direction of travel in the Southern Hemisphere (f < 0), which is
verified in `tests/test_coriolis.py`.

### Pressure gradient (`pressure_gradient.py`) — disabled by default

```
a_pressure ≈ -g * grad_h(SSH)
```

A simplified barotropic parameterization used only when a sea-surface-height
gradient field is actually available. It ignores baroclinic (density-driven)
structure below the surface. **Disabled by default** (`PhysicsConfig.pressure_gradient.enabled = False`)
because SSH/density fields are frequently unavailable, and this term is
never fabricated when they're missing.

### Sea-ice drag (`sea_ice_drag.py`)

```
F_si = 0.5 * rho_eff * C_Dsi * A_si * C_ice * |U_si - U_i| * (U_si - U_i)
```

An **empirical parameterization**, not a first-principles force — `C_Dsi`
has no agreed literature value and must be calibrated. If no explicit
sea-ice velocity product is available, `U_si` falls back to the ocean
surface current (`PhysicsConfig.sea_ice_drag.fallback_velocity_source`),
and this substitution is flagged via `EnvironmentalState.sea_ice_velocity_is_fallback`
— it is never presented as an observation.

### Grounding (`grounding.py`) — event-driven, not blended

Trigger: `D >= H_water(x, y)`, evaluated as a `solve_ivp` terminal event
(`numerical/events.py::make_grounding_event`). On grounding, the dynamics
mode switches to `GROUNDED` and free-floating forces (ocean/air/sea-ice/
pressure/Coriolis) are suppressed in favor of Coulomb friction:

```
F_friction = -mu * N * v / (|v| + epsilon)
N = max(0, (m - rho_w * V_sub) * g)     # "weight_minus_buoyancy" prototype model
```

This is an intentionally simple **prototype**. It does **not** model
sediment resistance, seabed slope, grounding geometry, or partial
grounding — those are documented extension points, not silent omissions.

## Melting (`melting.py`) — kept separate from the momentum equations

```
dH/dt = -M_basal
dL/dt = -(M_lateral + M_wave)
dW/dt = -(M_lateral + M_wave)
```

- **Basal melt**: bulk turbulent heat-transfer form,
  `M_basal = A + B * |U_rel|^C * (T_w - T_i)`, in the style of Bigg et al.
  (1997)-type Antarctic iceberg drift parameterizations.
- **Lateral melt**: linear in `(T_w - T_i)`.
- **Wave erosion**: proportional to wind speed above a threshold (a proxy
  for sea state).

All regression coefficients (`A, B, C, k_lateral, k_wave, wave_threshold_speed`)
are **UNVALIDATED DEFAULTS** — see `physics/melting.py::MeltRatesConfig`
docstrings. They make the module runnable out of the box; they are not
scientifically validated and must be calibrated per deployment.

## Numerical integration

Default solver: `RK45` (or `DOP853`), since the free-floating system is
not inherently stiff. `Radau` is available as an optional stiff solver
but is not assumed mandatory purely because grounding exists — grounding
is handled as event-driven hybrid dynamics (mode switching), not by
assuming the ODE itself is stiff.

Modes: `FREE_FLOATING`, `GROUNDED`, `MELTING_ONLY`, `TERMINATED`.
`numerical/deterministic_solver.py` runs the simulation in segments,
re-launching `solve_ivp` after each terminal event with the mode updated
and the event logged (`numerical/events.py::EventLog`).

## Stochastic extension

Euler-Maruyama discretization (`numerical/stochastic_solver.py`), kept
architecturally separate from the deterministic RK integrator — never a
deterministic step with noise bolted on afterward. Diffusion
(`uncertainty/diffusion.py`) starts as **diagonal, velocity-only** noise,
representing unresolved environmental/model forcing; position and
geometry are not perturbed without physical justification. Full
covariance is a documented future extension, not implemented by default.

## Residual model interface

`uncertainty/residuals.py::ResidualModel` is an abstract interface for a
future ConvLSTM (or other) residual predictor comparing `v_observed(t)`
against `v_physics(t)`. **No concrete implementation exists in this
codebase**, and nothing in `physics/*` imports or depends on it — the
physics engine is fully functional with `ResidualModel = None`, which is
the only configuration exercised by this task.

## Known limitations (a non-exhaustive list)

- Rectangular-block geometry is coarse; no keel roughness, tilt, or
  capsizing model.
- Grounding is a simple friction prototype; no sediment/slope/partial
  grounding.
- Sea-ice drag's contact-area model is a placeholder constant band width.
- All drag/melt coefficients are literature-typical defaults, not
  calibrated for any specific iceberg population.
- This is a **research/prototype decision-support system**. It carries no
  certification and makes no claim of guaranteed real-world navigational
  safety.
