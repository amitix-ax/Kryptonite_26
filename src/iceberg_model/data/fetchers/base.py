"""
Abstract interface every live data fetcher implements, so
`data/live_pipeline.py` can treat ERA5, AMSR2, BedMachine, and Copernicus
Marine uniformly.

A fetcher's job is strictly: given a BoundingBox + TimeRange, return the
raw field(s) as (lon, lat, time, values) -- reprojection onto the working
CRS happens uniformly afterward in live_pipeline.py via
data/reprojection.py, so no fetcher duplicates that logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from iceberg_model.data.fetchers.common import BoundingBox, TimeRange


@dataclass
class FetchedField:
    """
    Raw output of a fetcher, still in geographic (lon/lat) coordinates
    and native units, before reprojection.
    """

    variable_name: str
    lon: np.ndarray          # 1D, degrees
    lat: np.ndarray           # 1D, degrees
    values: np.ndarray         # shape (n_time, n_lat, n_lon) or (n_lat, n_lon) for static fields
    timestamps: Optional[np.ndarray] = None  # seconds since epoch, matches values.shape[0] if present
    units: str = ""
    source: str = ""


class DataFetcher(ABC):
    """One external data source. Subclasses must never fabricate data:
    if the source has no coverage for the requested bbox/time, raise
    rather than returning a plausible-looking but invented field."""

    source_name: str = "unknown"

    @abstractmethod
    def fetch(self, bbox: BoundingBox, time_range: TimeRange) -> FetchedField:
        raise NotImplementedError
