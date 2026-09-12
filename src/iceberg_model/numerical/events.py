"""
Event functions for scipy.integrate.solve_ivp (spec sections 21-22).

Grounding is treated as a genuine hybrid-dynamics event (root-finding on
keel_depth - water_depth), not folded into the continuous RHS. Each event
that fires is logged as a structured dict, matching the spec's example
schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.buoyancy import calculate_submerged_depth
from iceberg_model.physics.dynamics import EnvironmentalProvider
from iceberg_model.state.iceberg_state import IcebergState


@dataclass
class EventLog:
    events: list = field(default_factory=list)

    def record(self, time: float, event: str, **kwargs) -> None:
        entry = {"time": time, "event": event}
        entry.update(kwargs)
        self.events.append(entry)


def make_grounding_event(environmental_provider: EnvironmentalProvider, config: PhysicsConfig,
                          log: EventLog):
    """keel_depth - water_depth = 0, terminal, direction=+1 (approaching from below)."""

    def grounding_event(t: float, y: np.ndarray) -> float:
        state = IcebergState.from_vector(y, time=t)
        env = environmental_provider(state.x, state.y, t)
        if env.bathymetry is None or not env.validity_mask.get("bathymetry", True):
            # No bathymetry known here -> cannot evaluate; return a large
            # positive number so the event does not spuriously fire.
            return 1e9
        D = calculate_submerged_depth(state, config)
        return D - env.bathymetry

    grounding_event.terminal = True
    grounding_event.direction = 1
    return grounding_event


def make_geometry_invalid_event(min_dimension_m: float, log: EventLog):
    """Fires when min(L, W, H) drops to the configured minimum (melted out)."""

    def geometry_invalid_event(t: float, y: np.ndarray) -> float:
        _, _, _, _, L, W, H = y
        return min(L, W, H) - min_dimension_m

    geometry_invalid_event.terminal = True
    geometry_invalid_event.direction = -1
    return geometry_invalid_event


def make_dataset_boundary_event(bounds: tuple, log: EventLog):
    """Fires when the iceberg leaves the environmental dataset's spatial bounds.
    bounds = (left, bottom, right, top) in the working CRS."""
    left, bottom, right, top = bounds

    def dataset_boundary_event(t: float, y: np.ndarray) -> float:
        x, yy = y[0], y[1]
        dx = min(x - left, right - x)
        dy = min(yy - bottom, top - yy)
        return min(dx, dy)

    dataset_boundary_event.terminal = True
    dataset_boundary_event.direction = -1
    return dataset_boundary_event


def make_simulation_end_event(t_end: float, log: EventLog):
    def simulation_end_event(t: float, y: np.ndarray) -> float:
        return t_end - t

    simulation_end_event.terminal = True
    simulation_end_event.direction = -1
    return simulation_end_event
