"""
Temporal interpolation across discrete environmental snapshots
(spec section 10).

    alpha = (t - t0) / (t1 - t0)
    E(t) = (1-alpha) * E(t0) + alpha * E(t1)

Extrapolation is rejected by default. If the requested time lies outside
the dataset's time range, a TimeOutOfBoundsError is raised UNLESS
`allow_extrapolation` is explicitly enabled in
PhysicsConfig.temporal_interpolation -- environmental forcing is never
silently extrapolated.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class TimeOutOfBoundsError(Exception):
    pass


@dataclass
class TemporalField:
    """A time series of 2D (or ND) fields sharing one spatial grid."""
    timestamps: Sequence[float]   # seconds, strictly increasing
    fields: Sequence[np.ndarray]  # same length as timestamps

    def __post_init__(self):
        if len(self.timestamps) != len(self.fields):
            raise ValueError("timestamps and fields must have equal length")
        if list(self.timestamps) != sorted(self.timestamps):
            raise ValueError("timestamps must be strictly increasing")


def interpolate_in_time(tf: TemporalField, t: float, allow_extrapolation: bool = False) -> np.ndarray:
    ts = tf.timestamps
    n = len(ts)

    if n == 0:
        raise TimeOutOfBoundsError("TemporalField has no data.")

    if t <= ts[0]:
        if t == ts[0] or allow_extrapolation:
            return tf.fields[0]
        raise TimeOutOfBoundsError(
            f"Requested time {t} is before the earliest available timestamp {ts[0]} "
            "and extrapolation is not enabled."
        )
    if t >= ts[-1]:
        if t == ts[-1] or allow_extrapolation:
            return tf.fields[-1]
        raise TimeOutOfBoundsError(
            f"Requested time {t} is after the latest available timestamp {ts[-1]} "
            "and extrapolation is not enabled."
        )

    idx = bisect.bisect_right(ts, t) - 1
    t0, t1 = ts[idx], ts[idx + 1]
    e0, e1 = tf.fields[idx], tf.fields[idx + 1]

    alpha = (t - t0) / (t1 - t0)
    return (1 - alpha) * e0 + alpha * e1
