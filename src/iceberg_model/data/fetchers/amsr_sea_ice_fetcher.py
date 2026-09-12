"""
AMSR2 daily sea-ice concentration via NASA Earthdata (NSIDC-hosted).

Uses the real `earthaccess` client (`earthaccess.login()`,
`earthaccess.search_data()`, `earthaccess.download()`) against NASA's CMR
(Common Metadata Repository) -- not a stub. Requires a free NASA Earthdata
account (see credentials.py). Outbound access to urs.earthdata.nasa.gov /
n5eil01u.ecs.nsidc.org is required and is NOT available from this
development sandbox -- see data/live_pipeline.py's module note.

Dataset: AU_SI25 (AMSR2/GCOM-W1 daily 25 km sea ice concentration),
short_name "AU_SI25", NSIDC DAAC. https://nsidc.org/data/au_si25
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from iceberg_model.data.fetchers.base import DataFetcher, FetchedField
from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.credentials import EarthdataCredentials

logger = logging.getLogger(__name__)


class AMSRSeaIceFetcher(DataFetcher):
    source_name = "amsr2_sea_ice_concentration"
    short_name = "AU_SI25"

    def __init__(self, cache: DiskCache | None = None):
        self.cache = cache or DiskCache()

    def _login(self):
        import earthaccess
        # Never fabricate credentials -- fail loudly if none are configured,
        # matching every other fetcher's behavior.
        EarthdataCredentials.from_env()
        return earthaccess.login(strategy="environment")

    def fetch(self, bbox: BoundingBox, time_range: TimeRange) -> FetchedField:
        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError("xarray is required to read the downloaded AMSR2 granule.") from e

        self._login()
        import earthaccess

        granules = earthaccess.search_data(
            short_name=self.short_name,
            bounding_box=(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
            temporal=(time_range.start.isoformat(), time_range.end.isoformat()),
        )
        if not granules:
            logger.warning(
                "No %s granules found for bbox=%s, time_range=%s. "
                "Attempting ERA5 sea_ice_cover fallback before "
                "open-water assumption.",
                self.short_name, bbox, time_range,
            )
            return self._try_era5_fallback(bbox, time_range)
        logger.info("Found %d %s granule(s) for bbox=%s time_range=%s", len(granules), self.short_name, bbox, time_range)

        local_dir = self.cache.cache_dir / self.source_name
        local_dir.mkdir(parents=True, exist_ok=True)
        paths = earthaccess.download(granules, local_path=str(local_dir))
        logger.info("Downloaded %d granule(s) to %s", len(paths), local_dir)

        all_lon, all_lat, all_sic, all_ts = [], [], [], []
        for p in paths:
            ds = xr.open_dataset(p, group=None)
            # AU_SI25 provides separate polar-stereographic gridded
            # variables per hemisphere; the exact variable name follows
            # NSIDC's published schema (e.g. "SI_25km_SH_ICECON_DAY").
            sic_vars = [v for v in ds.data_vars if "ICECON" in v.upper()]
            if not sic_vars:
                ds.close()
                continue
            sic = ds[sic_vars[0]].values.astype(float)
            sic = np.where(sic > 100, np.nan, sic) / 100.0  # percent -> fraction; mask fill values
            lon = ds["longitude"].values if "longitude" in ds else ds["lon"].values
            lat = ds["latitude"].values if "latitude" in ds else ds["lat"].values
            all_lon.append(lon)
            all_lat.append(lat)
            all_sic.append(sic)
            all_ts.append(np.datetime64(ds.attrs.get("time_coverage_start", time_range.start.isoformat())))
            ds.close()

        if not all_sic:
            raise RuntimeError(
                f"Downloaded {len(paths)} {self.short_name} granule(s) but "
                "none contained a recognizable ICECON variable -- refusing "
                "to guess; inspect the granule schema and update the "
                "variable-name matching in this fetcher."
            )

        return FetchedField(
            variable_name="sea_ice_concentration",
            lon=all_lon[0], lat=all_lat[0],
            values=np.stack(all_sic, axis=0),
            timestamps=np.array(all_ts).astype("datetime64[s]").astype("int64"),
            units="fraction (0-1)",
            source="AMSR2/GCOM-W1 AU_SI25 (NSIDC DAAC via NASA Earthdata)",
        )

    def _try_era5_fallback(self, bbox: BoundingBox, time_range: TimeRange) -> FetchedField:
        """Attempt ERA5 sea_ice_cover as a secondary source before
        falling back to the open-water assumption (SIC=0)."""
        try:
            from iceberg_model.data.fetchers.era5_sea_ice_fetcher import ERA5SeaIceFetcher
            era5_sic = ERA5SeaIceFetcher(cache=self.cache).fetch(bbox, time_range)
            logger.info(
                "ERA5 sea_ice_cover fallback succeeded for bbox=%s, "
                "time_range=%s. Using reanalysis SIC instead of AMSR2.",
                bbox, time_range,
            )
            return era5_sic
        except Exception as e:
            logger.warning(
                "ERA5 sea_ice_cover fallback also failed (%s: %s). "
                "Falling back to open-water assumption (SIC=0). "
                "Sea-ice drag will be ZERO — trajectories in ice-covered "
                "waters will be unreliable.",
                type(e).__name__, e,
            )
            return self._fallback_open_water(bbox, time_range)

    def _fallback_open_water(self, bbox: BoundingBox, time_range: TimeRange) -> FetchedField:
        """Return a uniform SIC=0 (open water) field when real data is unavailable."""
        # Build a coarse 0.25° grid covering the bbox (matches ERA5 resolution)
        lon = np.arange(bbox.min_lon, bbox.max_lon + 0.25, 0.25)
        lat = np.arange(bbox.min_lat, bbox.max_lat + 0.25, 0.25)
        sic = np.zeros((1, len(lat), len(lon)), dtype=np.float64)
        ts = np.array([np.datetime64(time_range.start.isoformat())]).astype("datetime64[s]").astype("int64")
        return FetchedField(
            variable_name="sea_ice_concentration",
            lon=lon, lat=lat,
            values=sic,
            timestamps=ts,
            units="fraction (0-1)",
            source="Synthetic open-water fallback (no AMSR2 data available)",
        )
