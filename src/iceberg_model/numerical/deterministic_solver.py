"""
Deterministic solver (spec section 21).

Default integrator: RK45 or DOP853 (freely-floating dynamics are not
inherently stiff). Radau is available as an optional stiff solver, but is
NOT mandated just because grounding exists -- grounding is handled as a
hybrid-dynamics mode switch via event detection (numerical/events.py), not
by assuming stiffness.

The solver runs in segments: integrate until an event fires (grounding,
geometry-invalid, dataset-boundary, or simulation-end), switch
DynamicsMode accordingly, log the event, and re-launch solve_ivp from the
new state/mode. This continues until TERMINATED or simulation_end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from iceberg_model.config.physics_config import PhysicsConfig, SolverConfig
from iceberg_model.physics.dynamics import EnvironmentalProvider, dynamics
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.numerical.events import (
    EventLog,
    make_dataset_boundary_event,
    make_geometry_invalid_event,
    make_grounding_event,
    make_simulation_end_event,
)
from iceberg_model.state.iceberg_state import DynamicsMode, IcebergState


@dataclass
class TrajectorySegment:
    times: np.ndarray
    states: np.ndarray  # shape (n, 7)
    mode: DynamicsMode
    force_breakdowns: list = field(default_factory=list)


@dataclass
class SimulationResult:
    segments: list  # list[TrajectorySegment]
    event_log: EventLog
    final_mode: DynamicsMode

    def concatenated(self) -> tuple[np.ndarray, np.ndarray]:
        times = np.concatenate([s.times for s in self.segments])
        states = np.concatenate([s.states for s in self.segments], axis=0)
        return times, states


def run_deterministic_simulation(
    initial_state: IcebergState,
    t_end: float,
    environmental_provider: EnvironmentalProvider,
    config: PhysicsConfig,
    melt_rates_config: Optional[MeltRatesConfig] = None,
    dataset_bounds: Optional[tuple] = None,
    max_segments: int = 200,
) -> SimulationResult:
    """
    Run the deterministic physics engine from initial_state.time to t_end,
    handling grounding / geometry-invalid / dataset-boundary events as mode
    transitions. Returns a SimulationResult with one TrajectorySegment per
    dynamics-mode interval.
    """
    melt_rates_config = melt_rates_config or MeltRatesConfig()
    event_log = EventLog()
    segments: list[TrajectorySegment] = []

    solver_cfg: SolverConfig = config.solver
    method = solver_cfg.kind.value if hasattr(solver_cfg.kind, "value") else str(solver_cfg.kind)

    state = initial_state
    mode = initial_state.mode
    t_current = initial_state.time

    for _ in range(max_segments):
        if mode == DynamicsMode.TERMINATED or t_current >= t_end:
            break

        events = [make_simulation_end_event(t_end, event_log)]
        if mode == DynamicsMode.FREE_FLOATING and config.grounding.enabled:
            events.append(make_grounding_event(environmental_provider, config, event_log))
        if config.melting.enabled:
            events.append(make_geometry_invalid_event(config.melting.min_dimension_m, event_log))
        if dataset_bounds is not None:
            events.append(make_dataset_boundary_event(dataset_bounds, event_log))

        last_breakdown_holder: list = []

        def rhs(t, y, _mode=mode, _hold=last_breakdown_holder):
            return dynamics(t, y, environmental_provider, config, melt_rates_config,
                             mode=_mode, last_breakdown=_hold)

        sol = solve_ivp(
            rhs,
            t_span=(t_current, t_end),
            y0=state.as_vector(),
            method=method,
            rtol=solver_cfg.rtol,
            atol=solver_cfg.atol,
            max_step=solver_cfg.max_step_seconds or np.inf,
            events=events,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        segments.append(TrajectorySegment(times=sol.t, states=sol.y.T, mode=mode))

        t_current = sol.t[-1]
        state = IcebergState.from_vector(sol.y[:, -1], time=t_current, mode=mode)

        if t_current >= t_end - 1e-9:
            break

        # Determine which event fired (if any) to decide the next mode.
        fired = [i for i, te in enumerate(sol.t_events) if len(te) > 0]
        if not fired:
            break

        # Simulation-end event is always events[0].
        if 0 in fired:
            break

        # Identify grounding / geometry / boundary by position in `events`
        # list constructed above (order matches append order).
        idx_cursor = 1
        grounding_idx = geometry_idx = boundary_idx = None
        if mode == DynamicsMode.FREE_FLOATING and config.grounding.enabled:
            grounding_idx = idx_cursor
            idx_cursor += 1
        if config.melting.enabled:
            geometry_idx = idx_cursor
            idx_cursor += 1
        if dataset_bounds is not None:
            boundary_idx = idx_cursor
            idx_cursor += 1

        if grounding_idx is not None and grounding_idx in fired:
            event_log.record(t_current, "GROUNDING", x=state.x, y=state.y)
            mode = DynamicsMode.GROUNDED
        elif geometry_idx is not None and geometry_idx in fired:
            event_log.record(t_current, "GEOMETRY_INVALID", x=state.x, y=state.y,
                              L=state.L, W=state.W, H=state.H)
            mode = DynamicsMode.TERMINATED
        elif boundary_idx is not None and boundary_idx in fired:
            event_log.record(t_current, "DATASET_BOUNDARY", x=state.x, y=state.y)
            mode = DynamicsMode.TERMINATED
        else:
            break

        state.mode = mode

    final_mode = mode
    return SimulationResult(segments=segments, event_log=event_log, final_mode=final_mode)
