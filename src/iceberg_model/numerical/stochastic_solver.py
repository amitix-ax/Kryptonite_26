"""
Stochastic extension (spec sections 23-24): Euler-Maruyama discretization
of ds_t = f_phys(s_t, X_t) dt + G(s_t, X_t) dW_t.

    s_(t+dt) = s_t + f(s_t) dt + G(s_t) * sqrt(dt) * epsilon,   epsilon ~ N(0, I)

This module is deliberately kept SEPARATE from deterministic_solver.py:
it does not reuse solve_ivp's deterministic RK machinery with noise bolted
on afterward (which would silently misrepresent an Itô SDE).

Grounding and melting-termination are supported as fixed-step mode checks
(rather than the deterministic solver's continuous root-finding events,
which don't have a natural analog on a fixed Euler-Maruyama grid): each
step samples bathymetry at the current position and checks
`is_grounded()`/geometry validity BEFORE advancing, switching
DynamicsMode and logging the transition exactly once. Velocity noise is
only injected while FREE_FLOATING -- once GROUNDED or TERMINATED,
Coulomb friction (GROUNDED) or nothing (TERMINATED) governs the state,
and adding stochastic noise on top would not represent unresolved ocean/
wind forcing (there is no free-floating forcing left to be uncertain
about).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.dynamics import EnvironmentalProvider, dynamics
from iceberg_model.physics.grounding import is_grounded
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.state.iceberg_state import DynamicsMode, IcebergState
from iceberg_model.uncertainty.diffusion import DiffusionConfig, diagonal_velocity_diffusion_matrix


@dataclass
class StochasticTrajectory:
    times: np.ndarray
    states: np.ndarray            # (n, 7)
    modes: list = field(default_factory=list)   # DynamicsMode per step, len == len(times)
    events: list = field(default_factory=list)  # [{"time":..., "event": "GROUNDING"/"GEOMETRY_INVALID", ...}]


def run_euler_maruyama(
    initial_state: IcebergState,
    t_end: float,
    dt: float,
    environmental_provider: EnvironmentalProvider,
    config: PhysicsConfig,
    diffusion_config: DiffusionConfig,
    melt_rates_config: Optional[MeltRatesConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> StochasticTrajectory:
    melt_rates_config = melt_rates_config or MeltRatesConfig()
    rng = rng or np.random.default_rng()

    G = diagonal_velocity_diffusion_matrix(diffusion_config)
    sqrt_dt = np.sqrt(dt)

    n_steps = int(np.ceil((t_end - initial_state.time) / dt))
    times = np.empty(n_steps + 1)
    states = np.empty((n_steps + 1, 7))
    modes: list = [None] * (n_steps + 1)
    events: list = []

    s = initial_state.as_vector().copy()
    t = initial_state.time
    mode = initial_state.mode

    times[0], states[0], modes[0] = t, s, mode

    for i in range(1, n_steps + 1):
        state_now = IcebergState.from_vector(s, time=t, mode=mode)

        if mode == DynamicsMode.FREE_FLOATING:
            if config.grounding.enabled:
                env_now = environmental_provider(state_now.x, state_now.y, t)
                if env_now.bathymetry is not None and env_now.validity_mask.get("bathymetry", True):
                    if is_grounded(state_now, config, env_now.bathymetry):
                        mode = DynamicsMode.GROUNDED
                        events.append({"time": t, "event": "GROUNDING", "x": state_now.x, "y": state_now.y})
            if config.melting.enabled and not state_now.is_geometrically_valid(config.melting.min_dimension_m):
                mode = DynamicsMode.TERMINATED
                events.append({"time": t, "event": "GEOMETRY_INVALID", "x": state_now.x, "y": state_now.y})

        drift = dynamics(t, s, environmental_provider, config, melt_rates_config, mode=mode)

        # Noise only while genuinely free-floating -- see module docstring.
        inject_noise = diffusion_config.enabled and mode == DynamicsMode.FREE_FLOATING
        epsilon = rng.normal(size=7) if inject_noise else np.zeros(7)

        s = s + drift * dt + (G @ epsilon) * sqrt_dt
        t = t + dt

        times[i], states[i], modes[i] = t, s, mode

    # Catch a mode transition that only becomes true exactly at the final
    # step's output (there's no subsequent iteration to detect it in the
    # loop above) -- otherwise a trajectory that melts out or grounds on
    # its very last step would be reported as still FREE_FLOATING.
    final_state = IcebergState.from_vector(states[-1], time=times[-1], mode=mode)
    if mode == DynamicsMode.FREE_FLOATING:
        if config.grounding.enabled:
            env_final = environmental_provider(final_state.x, final_state.y, times[-1])
            if env_final.bathymetry is not None and env_final.validity_mask.get("bathymetry", True):
                if is_grounded(final_state, config, env_final.bathymetry):
                    mode = DynamicsMode.GROUNDED
                    events.append({"time": times[-1], "event": "GROUNDING",
                                   "x": final_state.x, "y": final_state.y})
        if mode == DynamicsMode.FREE_FLOATING and config.melting.enabled and \
                not final_state.is_geometrically_valid(config.melting.min_dimension_m):
            mode = DynamicsMode.TERMINATED
            events.append({"time": times[-1], "event": "GEOMETRY_INVALID",
                            "x": final_state.x, "y": final_state.y})
        modes[-1] = mode

    return StochasticTrajectory(times=times, states=states, modes=modes, events=events)
