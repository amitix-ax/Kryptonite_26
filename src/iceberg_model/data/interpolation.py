"""
Spatial interpolation of a gridded field at an arbitrary (x, y) point
on the common working grid.

Every interpolation call is traceable (principle I): callers get back not
just a value but a validity flag, so a NaN/masked/out-of-bounds query
never silently becomes a valid physical value (principle J).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class InterpolationResult:
    value: Optional[float]
    valid: bool
    reason: str = ""  # populated when valid=False, e.g. "out_of_bounds", "nodata"


def bilinear_interpolate(field: np.ndarray, transform: tuple, x: float, y: float) -> InterpolationResult:
    """
    Bilinear-interpolate `field` (2D array on a regular grid defined by
    `transform` = (xres, 0, x_origin, 0, -yres, y_origin), matching
    reprojection.CommonGrid.transform) at projected coordinate (x, y).
    """
    xres, _, x0, _, yres, y0 = transform  # yres is negative (north-up)
    col = (x - x0) / xres
    row = (y - y0) / yres

    nrows, ncols = field.shape[-2], field.shape[-1]

    if col < 0 or row < 0 or col > ncols - 1 or row > nrows - 1:
        return InterpolationResult(value=None, valid=False, reason="out_of_bounds")

    c0, r0 = int(np.floor(col)), int(np.floor(row))
    c1, r1 = min(c0 + 1, ncols - 1), min(r0 + 1, nrows - 1)
    fc, fr = col - c0, row - r0

    v00 = field[r0, c0]
    v01 = field[r0, c1]
    v10 = field[r1, c0]
    v11 = field[r1, c1]

    vals = np.ma.array([v00, v01, v10, v11], dtype=float)
    if np.any(np.isnan(np.ma.filled(vals, np.nan))) or np.ma.is_masked(vals):
        return InterpolationResult(value=None, valid=False, reason="nodata")

    top = v00 * (1 - fc) + v01 * fc
    bot = v10 * (1 - fc) + v11 * fc
    value = top * (1 - fr) + bot * fr

    return InterpolationResult(value=float(value), valid=True)


def nearest_interpolate(field: np.ndarray, transform: tuple, x: float, y: float) -> InterpolationResult:
    xres, _, x0, _, yres, y0 = transform
    col = int(round((x - x0) / xres))
    row = int(round((y - y0) / yres))
    nrows, ncols = field.shape[-2], field.shape[-1]

    if col < 0 or row < 0 or col >= ncols or row >= nrows:
        return InterpolationResult(value=None, valid=False, reason="out_of_bounds")

    val = field[row, col]
    if np.isnan(val) or np.ma.is_masked(val):
        return InterpolationResult(value=None, valid=False, reason="nodata")

    return InterpolationResult(value=float(val), valid=True)
