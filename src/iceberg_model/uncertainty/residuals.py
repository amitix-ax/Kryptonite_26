"""
Residual-model interface (spec section 25): prepares the physics engine
for future ConvLSTM residual correction WITHOUT requiring it.

    v_physics(t)   -- velocity from the physics engine alone
    v_observed(t)  -- observed velocity (e.g. from satellite tracking)
    r(t) = v_observed(t) - v_physics(t)

`ResidualModel` is an abstract interface. The physics engine (dynamics.py,
deterministic_solver.py) never imports a concrete ResidualModel and works
correctly with `ResidualModel = None` -- this module exists purely so a
future ConvLSTM (or any other residual predictor) can be plugged in at the
simulation layer (simulation/simulator.py) without touching physics/*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from iceberg_model.state.iceberg_state import ForceBreakdown


@dataclass
class ResidualDistribution:
    """A predicted residual velocity correction with uncertainty."""
    mean: np.ndarray        # [dr_u, dr_v] m/s
    covariance: Optional[np.ndarray] = None  # 2x2, or None if point-estimate only


class ResidualModel(ABC):
    """Abstract interface a future ConvLSTM (or other) residual predictor
    must implement. NOT called anywhere in physics/*."""

    @abstractmethod
    def predict_distribution(self, environment_features: dict,
                              physics_state: dict) -> ResidualDistribution:
        """
        environment_features: arbitrary dict of environmental context
            (e.g. local SIC field patch, ocean current field patch, etc.)
            -- the exact schema is intentionally left to the future model.
        physics_state: dict view of IcebergState + ForceBreakdown, giving
            the residual model visibility into which physical mechanism
            (ocean/air/coriolis/pressure/sea_ice/grounding) is currently
            dominant, per the ForceBreakdown explainability requirement.
        """
        raise NotImplementedError


def compute_observed_residual(v_observed: np.ndarray, v_physics: np.ndarray) -> np.ndarray:
    """r(t) = v_observed(t) - v_physics(t). Used for offline residual-model
    training/validation, not inside the online physics loop."""
    return np.asarray(v_observed) - np.asarray(v_physics)


def force_breakdown_to_state_dict(breakdown: ForceBreakdown) -> dict:
    """Convenience: expose the physics-engine's force decomposition as a
    plain dict for a future ResidualModel's `physics_state` argument."""
    return {
        "ocean": breakdown.ocean.tolist(),
        "air": breakdown.air.tolist(),
        "coriolis": breakdown.coriolis.tolist(),
        "pressure": breakdown.pressure.tolist(),
        "sea_ice": breakdown.sea_ice.tolist(),
        "grounding": breakdown.grounding.tolist(),
        "total": breakdown.total.tolist(),
    }
