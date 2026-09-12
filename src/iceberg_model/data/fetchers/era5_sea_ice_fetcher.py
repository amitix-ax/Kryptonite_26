"""
ERA5 sea-ice concentration (``sea_ice_cover``) via the Copernicus Climate
Data Store's ``cdsapi`` client.

This serves as a **fallback** when the primary AMSR2 AU_SI25 data from
NSIDC is unavailable (EULA not accepted, temporal gap, network issues).
ERA5 provides sea-ice concentration as a 0–1 fraction on the same 0.25°
global grid already used for wind (``era5_fetcher.py``), and shares the
same CDS API credentials — no additional account setup required.

Dataset: ``reanalysis-era5-single-levels``, variable ``sea_ice_cover``.
Docs: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from iceberg_model.data.fetchers.base import DataFetcher, FetchedField
from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.credentials import CDSCredentials

logger = logging.getLogger(__name__)


class ERA5SeaIceFetcher(DataFetcher):
    """Fetch sea-ice concentration from ERA5 reanalysis via the CDS API.

    Returns the ``sea_ice_cover`` variable (already 0–1 fraction, no
    rescaling needed) as a :class:`FetchedField` with
    ``variable_name="sea_ice_concentration"`` so it is a drop-in
    replacement for AMSR2 SIC anywhere in the pipeline.
    """

    source_name = "era5_sea_ice"

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
            raise ImportError(
                "xarray is required to read the downloaded ERA5 NetCDF file."
            ) from e

        params = {
            "min_lon": bbox.min_lon, "max_lon": bbox.max_lon,
            "min_lat": bbox.min_lat, "max_lat": bbox.max_lat,
            "start": time_range.start, "end": time_range.end,
        }
        target = self.cache.path_for(self.source_name, params, suffix=".nc")

        if not self.cache.is_fresh(target):
            logger.info(
                "Fetching ERA5 sea_ice_cover for bbox=%s time_range=%s (cache miss)",
                bbox, time_range,
            )
            client = self._build_client()
            # CDS area format: [North, West, South, East]
            area = [bbox.max_lat, bbox.min_lon, bbox.min_lat, bbox.max_lon]
            days = sorted({d.strftime("%d") for d in _date_range(time_range)})
            months = sorted({d.strftime("%m") for d in _date_range(time_range)})
            years = sorted({d.strftime("%Y") for d in _date_range(time_range)})

            request = {
                "product_type": "reanalysis",
                "variable": ["sea_ice_cover"],
                "year": years,
                "month": months,
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": area,
                "format": "netcdf",
            }
            client.retrieve("reanalysis-era5-single-levels", request, str(target))
            logger.info("ERA5 sea_ice_cover downloaded to %s", target)
        else:
            logger.info("Using cached ERA5 sea_ice_cover at %s", target)

        ds = xr.open_dataset(target)
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        # ERA5 sea_ice_cover is already 0–1 fraction; no rescaling needed.
        sic = ds["siconc"].values if "siconc" in ds else ds["sea_ice_cover"].values
        # Newer CDS API returns 'valid_time' instead of 'time'
        time_var = "valid_time" if "valid_time" in ds else "time"
        timestamps = ds[time_var].values.astype("datetime64[s]").astype("int64")
        ds.close()

        # Clamp to [0, 1] to guard against any fill-value leakage.
        sic = np.clip(sic, 0.0, 1.0)

        return FetchedField(
            variable_name="sea_ice_concentration",
            lon=lon,
            lat=lat,
            values=sic,
            timestamps=timestamps,
            units="fraction (0-1)",
            source="ERA5 reanalysis sea_ice_cover (Copernicus CDS) [fallback]",
        )


def _date_range(time_range: TimeRange):
    from datetime import timedelta
    d = time_range.start
    while d <= time_range.end:
        yield d
        d += timedelta(days=1)
