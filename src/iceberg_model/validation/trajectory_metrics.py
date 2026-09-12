"""
Trajectory-comparison metrics for validating simulated drift against
observed iceberg tracks (e.g. from satellite-derived positions).
"""

from __future__ import annotations

import numpy as np


def position_error_series(sim_xy: np.ndarray, obs_xy: np.ndarray) -> np.ndarray:
    """Euclidean distance [m] between simulated and observed positions at
    matching timestamps. Both arrays shape (n, 2)."""
    sim_xy, obs_xy = np.asarray(sim_xy), np.asarray(obs_xy)
    if sim_xy.shape != obs_xy.shape:
        raise ValueError(f"Shape mismatch: sim {sim_xy.shape} vs obs {obs_xy.shape}")
    return np.linalg.norm(sim_xy - obs_xy, axis=1)


def cumulative_along_track_skill(sim_xy: np.ndarray, obs_xy: np.ndarray,
                                  persistence_xy: np.ndarray) -> float:
    """
    Skill score comparing physics-model position error to a naive
    persistence baseline (iceberg stays put): 1.0 = perfect, 0.0 = no
    better than persistence, negative = worse than persistence.
    """
    model_err = position_error_series(sim_xy, obs_xy)
    persistence_err = position_error_series(persistence_xy, obs_xy)
    denom = np.sum(persistence_err)
    if denom == 0:
        raise ValueError("Persistence baseline has zero cumulative error; skill score undefined.")
    return float(1.0 - np.sum(model_err) / denom)
