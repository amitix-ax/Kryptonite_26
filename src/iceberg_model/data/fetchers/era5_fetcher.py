"""
ERA5 10m wind components (u10, v10) via the Copernicus Climate Data
Store's `cdsapi` client.

This calls the REAL CDS API (`cdsapi.Client().retrieve(...)`) -- it is not
a stub. It requires:
    1. A free CDS account + API key (see credentials.py for setup).
    2. Outbound network access to cds.climate.copernicus.eu, which this
       development sandbox does NOT have -- see the module-level note in
       data/live_pipeline.py for why this can't be demonstrated live here.

Dataset: `reanalysis-era5-single-levels`, hourly, 0.25-degree resolution.
Docs: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from iceberg_model.data.fetchers.base import DataFetcher, FetchedField
from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.credentials import CDSCredentials

logger = logging.getLogger(__name__)


class ERA5WindFetcher(DataFetcher):
    source_name = "era5_wind"

    def __init__(self, cache: DiskCache | None = None):
        self.cache = cache or DiskCache()

    def _build_client(self):
        import cdsapi
        creds = CDSCredentials.from_env()
        return cdsapi.Client(url=creds.url, key=creds.key)

    def fetch(self, bbox: BoundingBox, time_range: TimeRange) -> FetchedField:
        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError("xarray is required to read the downloaded ERA5 NetCDF file.") from e

        params = {
            "min_lon": bbox.min_lon, "max_lon": bbox.max_lon,
            "min_lat": bbox.min_lat, "max_lat": bbox.max_lat,
            "start": time_range.start, "end": time_range.end,
        }
        target = self.cache.path_for(self.source_name, params, suffix=".nc")

        if not self.cache.is_fresh(target):
            logger.info("Fetching ERA5 wind for bbox=%s time_range=%s (cache miss)", bbox, time_range)
            client = self._build_client()
            # CDS area format: [North, West, South, East]
            area = [bbox.max_lat, bbox.min_lon, bbox.min_lat, bbox.max_lon]
            days = sorted({d.strftime("%d") for d in _date_range(time_range)})
            months = sorted({d.strftime("%m") for d in _date_range(time_range)})
            years = sorted({d.strftime("%Y") for d in _date_range(time_range)})

            request = {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": years,
                "month": months,
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": area,
                "format": "netcdf",
            }
            client.retrieve("reanalysis-era5-single-levels", request, str(target))
            logger.info("ERA5 wind downloaded to %s", target)
        else:
            logger.info("Using cached ERA5 wind at %s", target)

        ds = xr.open_dataset(target)
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        u10 = ds["u10"].values  # (time, lat, lon)
        v10 = ds["v10"].values
        # Newer CDS API returns 'valid_time' instead of 'time'
        time_var = "valid_time" if "valid_time" in ds else "time"
        timestamps = ds[time_var].values.astype("datetime64[s]").astype("int64")
        ds.close()

        return FetchedField(
            variable_name="wind_uv",
            lon=lon, lat=lat,
            values=np.stack([u10, v10], axis=0),  # (2, time, lat, lon): [u, v]
            timestamps=timestamps,
            units="m/s",
            source="ERA5 reanalysis-single-levels (Copernicus CDS)",
        )


def _date_range(time_range: TimeRange):
    from datetime import timedelta
    d = time_range.start
    while d <= time_range.end:
        yield d
        d += timedelta(days=1)
