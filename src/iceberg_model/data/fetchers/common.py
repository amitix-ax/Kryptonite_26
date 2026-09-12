"""
Shared request types and a simple disk cache used by every fetcher in
this package, so repeated runs (e.g. re-running a simulation over the
same region/day during development) don't re-hit external services.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box in EPSG:4326 (lon/lat degrees) -- the
    natural query interface for every external API used here. Reprojected
    onto the working CRS later, via data/reprojection.py, not before."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self):
        if not (-180 <= self.min_lon < self.max_lon <= 180):
            raise ValueError(f"Invalid longitude range: [{self.min_lon}, {self.max_lon}]")
        if not (-90 <= self.min_lat < self.max_lat <= 90):
            raise ValueError(f"Invalid latitude range: [{self.min_lat}, {self.max_lat}]")


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"TimeRange start {self.start} is after end {self.end}")


class DiskCache:
    """
    Minimal content-addressed disk cache: a fetch request (source name +
    parameters) hashes to a cache key; if a file already exists for that
    key, fetchers should return it instead of re-downloading.

    This is a cache, not a source of truth -- it never fabricates data,
    it only avoids re-fetching identical requests within `max_age_days`.
    """

    def __init__(self, cache_dir: str | Path = "data/cache", max_age_days: Optional[float] = 7.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days

    def key_for(self, source: str, params: dict) -> str:
        serializable = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in params.items()
        }
        blob = json.dumps({"source": source, "params": serializable}, sort_keys=True)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:24]
        return f"{source}_{digest}"

    def path_for(self, source: str, params: dict, suffix: str) -> Path:
        return self.cache_dir / f"{self.key_for(source, params)}{suffix}"

    def is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.max_age_days is None:
            return True
        age_seconds = datetime.now().timestamp() - path.stat().st_mtime
        return age_seconds < self.max_age_days * 86400
