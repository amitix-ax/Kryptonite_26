"""
Ocean surface current components (uo, vo) via the Copernicus Marine
Service (CMEMS) `copernicusmarine` client.

Uses the real `copernicusmarine.subset(...)` call -- not a stub. Requires
a free Copernicus Marine account (see credentials.py). Outbound access to
data.marine.copernicus.eu is required and is NOT available from this
development sandbox -- see data/live_pipeline.py's module note.

Dataset: GLOBAL_ANALYSISFORECAST_PHY_001_024 (or the equivalent
reanalysis product GLOBAL_MULTIYEAR_PHY_001_030 for dates outside the
forecast system's window), variables uo/vo at the surface depth level.
https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024
"""

from __future__ import annotations

import logging

import numpy as np

from iceberg_model.data.fetchers.base import DataFetcher, FetchedField
from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.credentials import CopernicusMarineCredentials

logger = logging.getLogger(__name__)


class CopernicusMarineCurrentFetcher(DataFetcher):
    source_name = "copernicus_marine_ocean_current"
    dataset_id_forecast = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
    dataset_id_reanalysis = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

    def __init__(self, cache: DiskCache | None = None, use_reanalysis: bool = False):
        self.cache = cache or DiskCache()
        self.use_reanalysis = use_reanalysis

    def fetch(self, bbox: BoundingBox, time_range: TimeRange) -> FetchedField:
        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError("xarray is required to read the downloaded ocean-current NetCDF file.") from e
        import copernicusmarine

        creds = CopernicusMarineCredentials.from_env()

        params = {
            "min_lon": bbox.min_lon, "max_lon": bbox.max_lon,
            "min_lat": bbox.min_lat, "max_lat": bbox.max_lat,
            "start": time_range.start, "end": time_range.end,
            "reanalysis": self.use_reanalysis,
        }
        target = self.cache.path_for(self.source_name, params, suffix=".nc")

        if not self.cache.is_fresh(target):
            dataset_id = self.dataset_id_reanalysis if self.use_reanalysis else self.dataset_id_forecast
            logger.info("Fetching Copernicus Marine ocean currents (%s) for bbox=%s time_range=%s (cache miss)",
                        dataset_id, bbox, time_range)
            copernicusmarine.subset(
                dataset_id=dataset_id,
                username=creds.username,
                password=creds.password,
                variables=["uo", "vo"],
                minimum_longitude=bbox.min_lon,
                maximum_longitude=bbox.max_lon,
                minimum_latitude=bbox.min_lat,
                maximum_latitude=bbox.max_lat,
                minimum_depth=0.0,
                maximum_depth=1.0,  # surface layer only for the depth-averaged drag model
                start_datetime=time_range.start,
                end_datetime=time_range.end,
                output_filename=target.name,
                output_directory=str(target.parent),
                overwrite=True,
            )
            logger.info("Copernicus Marine ocean currents downloaded to %s", target)
        else:
            logger.info("Using cached Copernicus Marine ocean currents at %s", target)

        ds = xr.open_dataset(target)
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        uo = ds["uo"].isel(depth=0).values if "depth" in ds["uo"].dims else ds["uo"].values
        vo = ds["vo"].isel(depth=0).values if "depth" in ds["vo"].dims else ds["vo"].values
        # Newer Copernicus Marine products may use 'valid_time' instead of 'time'
        time_var = "valid_time" if "valid_time" in ds else "time"
        timestamps = ds[time_var].values.astype("datetime64[s]").astype("int64")
        ds.close()

        return FetchedField(
            variable_name="ocean_uv",
            lon=lon, lat=lat,
            values=np.stack([uo, vo], axis=0),  # (2, time, lat, lon)
            timestamps=timestamps,
            units="m/s",
            source=f"Copernicus Marine Service ({'reanalysis' if self.use_reanalysis else 'analysis-forecast'})",
        )
