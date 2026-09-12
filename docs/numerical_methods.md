# Numerical Methods

## Deterministic solver

`numerical/deterministic_solver.py::run_deterministic_simulation()` wraps
`scipy.integrate.solve_ivp`. Default method is `RK45`; `DOP853` and
`Radau` are available via `PhysicsConfig.solver.kind`. `Radau` (implicit,
suited to stiff problems) is offered as an option, **not** assumed
necessary merely because grounding exists in the model — grounding is a
discrete mode switch (hybrid dynamics), handled via event detection, not
a source of continuous-ODE stiffness in the free-floating regime.

### Hybrid dynamics via events

The solver runs in segments. Each segment integrates the current
`DynamicsMode`'s RHS (`physics/dynamics.py::dynamics`) until one of these
terminal events fires (`numerical/events.py`):

- **Grounding**: `keel_depth - water_depth = 0`, direction `+1`
  (approaching grounding from above). Terminal.
- **Geometry invalid**: `min(L, W, H) - min_dimension = 0`, direction
  `-1` (iceberg has melted below the configured minimum size). Terminal.
- **Dataset boundary**: iceberg exits the environmental dataset's spatial
  bounds. Terminal.
- **Simulation end**: `t_end - t = 0`. Terminal.

When an event fires, `run_deterministic_simulation` logs it
(`EventLog.record`, matching the spec's `{time, event, x, y, ...}`
schema), updates `DynamicsMode` accordingly, and re-launches `solve_ivp`
from the current state. This continues until `TERMINATED` or `t_end` is
reached, or `max_segments` is exhausted (a safety cap against infinite
event loops).

### Convergence testing

`validation/metrics.py::richardson_convergence_order()` estimates the
empirical order of convergence from two runs at different step sizes,
assuming `error ~ C * h^p`. `tests/test_solver.py` exercises this against
both a synthetic ODE with a known analytic solution and the full
grounding-event pathway, so numerical behavior is testable, not just
assumed (spec principle F).

## Stochastic solver

`numerical/stochastic_solver.py::run_euler_maruyama()` implements a
fixed-step Euler-Maruyama discretization of:

```
ds_t = f_phys(s_t, X_t) dt + G(s_t, X_t) dW_t
s_(t+dt) = s_t + f(s_t) dt + G(s_t) * sqrt(dt) * epsilon,   epsilon ~ N(0, I)
```

This is a **separate code path** from the deterministic RK integrator —
it is not the deterministic solver with noise injected after the fact,
which would misrepresent the Itô SDE. It currently supports
`FREE_FLOATING` dynamics only; grounding/melting-event handling for the
stochastic path is a documented extension point.

`uncertainty/ensemble.py::run_ensemble()` runs N independent
Euler-Maruyama realizations to build a `TrajectoryEnsemble`, the input the
(out-of-scope-for-this-task) collision-probability / risk-map stages
would consume.

## Diffusion

`uncertainty/diffusion.py::DiffusionConfig` — diagonal, velocity-only by
default (`G` nonzero only on the `u, v` rows). This represents unresolved
environmental/model forcing uncertainty, not arbitrary noise on every
state variable; position and geometry remain deterministic. Non-diagonal
(full covariance) diffusion is a named-but-unimplemented extension.
