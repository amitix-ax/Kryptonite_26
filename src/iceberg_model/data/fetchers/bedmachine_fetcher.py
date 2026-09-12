"""
BedMachine Antarctica bathymetry/bed topography via NASA Earthdata
(NSIDC-hosted).

Uses the real `earthaccess` client, same as amsr_sea_ice_fetcher.py.
BedMachine is a single static (non-time-varying) product, so this
fetcher ignores TimeRange -- water depth doesn't have a temporal
dimension the way ocean current/wind/sea-ice do.

Dataset: NSIDC-0756 "MEaSUREs BedMachine Antarctica", short_name
"NSIDC-0756". https://nsidc.org/data/nsidc-0756
"""

from __future__ import annotations

import logging

import numpy as np

from iceberg_model.data.fetchers.base import DataFetcher, FetchedField
from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.credentials import EarthdataCredentials

logger = logging.getLogger(__name__)


class BedMachineBathymetryFetcher(DataFetcher):
    source_name = "bedmachine_antarctica_bathymetry"
    short_name = "NSIDC-0756"

    def __init__(self, cache: DiskCache | None = None):
        self.cache = cache or DiskCache(max_age_days=None)  # static product; cache indefinitely

    def _login(self):
        import earthaccess
        EarthdataCredentials.from_env()
        return earthaccess.login(strategy="environment")

    def fetch(self, bbox: BoundingBox, time_range: TimeRange | None = None) -> FetchedField:
        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError("xarray is required to read the BedMachine NetCDF file.") from e

        self._login()
        import earthaccess

        granules = earthaccess.search_data(short_name=self.short_name, count=1)
        if not granules:
            raise RuntimeError(
                f"No {self.short_name} granules found via CMR. BedMachine "
                "Antarctica ships as a single full-continent file, so an "
                "empty result usually indicates an authentication or "
                "short_name/version mismatch, not a coverage gap."
            )

        local_dir = self.cache.cache_dir / self.source_name
        local_dir.mkdir(parents=True, exist_ok=True)
        paths = earthaccess.download(granules, local_path=str(local_dir))
        logger.info("BedMachine Antarctica downloaded/cached at %s", paths[0])

        ds = xr.open_dataset(paths[0])
        # BedMachine ships on its native EPSG:3031 polar-stereographic
        # grid with 'x'/'y' (meters), NOT lon/lat -- unlike the other
        # fetchers, so we keep it in that native projection and let
        # live_pipeline.py reproject directly from EPSG:3031 rather than
        # forcing an unnecessary lon/lat round-trip.
        bed = ds["bed"].values          # bed topography, meters relative to sea level (negative = below sea level)
        surface = ds["surface"].values if "surface" in ds else None
        x = ds["x"].values
        y = ds["y"].values
        ds.close()

        # Bathymetry (positive-down water depth) = -bed, clipped to >= 0
        # (bed above sea level is land, not a valid ocean depth here).
        water_depth = np.clip(-bed, 0.0, None)

        return FetchedField(
            variable_name="bathymetry",
            lon=x,  # native EPSG:3031 meters, NOT degrees -- see docstring above
            lat=y,
            values=water_depth,
            timestamps=None,
            units="m (positive-down water depth, derived from BedMachine 'bed')",
            source="MEaSUREs BedMachine Antarctica NSIDC-0756 (NSIDC DAAC via NASA Earthdata)",
        )
