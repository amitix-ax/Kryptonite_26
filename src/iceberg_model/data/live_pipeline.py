"""
Automatic environment builder: the single entry point that replaces
manually constructing an EnvironmentalDataset (as
examples/synthetic_environment.py does) with one that pulls real data
directly from each source's live API.

    build_live_environment(bbox, time_range, grid)
        -> fetches ERA5 wind, AMSR2 sea-ice concentration, Copernicus
           Marine ocean currents, and BedMachine Antarctica bathymetry
        -> reprojects each onto the same CommonGrid
        -> returns an EnvironmentalDataset ready to pass to Simulator

IMPORTANT — sandbox network limitation:
This development environment's outbound network access is restricted to
package registries (PyPI, GitHub, npm) for security reasons. It cannot
reach cds.climate.copernicus.eu, urs.earthdata.nasa.gov, or
data.marine.copernicus.eu, so this module's live-fetch path cannot be
executed *from this sandbox*. Every fetcher underneath it
(era5_fetcher.py, amsr_sea_ice_fetcher.py, bedmachine_fetcher.py,
ocean_current_fetcher.py) calls the real client library for its service
(cdsapi, earthaccess, copernicusmarine) with correct request parameters,
and will run as soon as this code is deployed somewhere with open network
access and the credentials described in fetchers/credentials.py.

`examples/fetch_and_run_live.py` demonstrates the intended call pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from iceberg_model.data.environmental_dataset import EnvironmentalDataset
from iceberg_model.data.fetchers.amsr_sea_ice_fetcher import AMSRSeaIceFetcher
from iceberg_model.data.fetchers.base import FetchedField
from iceberg_model.data.fetchers.bedmachine_fetcher import BedMachineBathymetryFetcher
from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.era5_fetcher import ERA5WindFetcher
from iceberg_model.data.fetchers.ocean_current_fetcher import CopernicusMarineCurrentFetcher
from iceberg_model.data.reprojection import CommonGrid, ResamplingKind, VARIABLE_RESAMPLING_DEFAULTS
from iceberg_model.data.temporal_loader import TemporalField


def _lonlat_affine(lon: np.ndarray, lat: np.ndarray):
    """Build a north-up affine transform tuple for a regular lon/lat grid,
    for use as the *source* transform in a reprojection call. Assumes
    uniform spacing (true for ERA5/AMSR2/CMEMS gridded products)."""
    dx = float(lon[1] - lon[0])
    dy = float(lat[1] - lat[0])
    west = float(lon[0]) - dx / 2
    # rasterio wants north-up (negative row pitch); flip if lat is ascending.
    if dy > 0:
        north = float(lat[-1]) + dy / 2
        row_pitch = -dy
    else:
        north = float(lat[0]) - dy / 2
        row_pitch = dy
    return (dx, 0.0, west, 0.0, row_pitch, north)


def _reproject_2d(values: np.ndarray, src_transform: tuple, src_crs: str,
                   grid: CommonGrid, resampling: ResamplingKind) -> np.ndarray:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine

    resampling_map = {
        ResamplingKind.NEAREST: Resampling.nearest,
        ResamplingKind.BILINEAR: Resampling.bilinear,
        ResamplingKind.CUBIC: Resampling.cubic,
    }
    dst = np.full((grid.height, grid.width), np.nan, dtype=np.float64)
    reproject(
        source=values,
        destination=dst,
        src_transform=Affine(*src_transform),
        src_crs=src_crs,
        dst_transform=Affine(*grid.transform),
        dst_crs=grid.crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling_map[resampling],
    )
    return dst


def _field_to_temporal_field(field: FetchedField, src_crs: str, grid: CommonGrid,
                              variable_key: str) -> TemporalField:
    transform = _lonlat_affine(field.lon, field.lat) if src_crs == "EPSG:4326" else grid.transform
    resampling = VARIABLE_RESAMPLING_DEFAULTS.get(variable_key, ResamplingKind.BILINEAR)

    if field.timestamps is None:
        raise ValueError(f"{field.variable_name} has no timestamps; use a static field instead.")

    reprojected = [
        _reproject_2d(field.values[i], transform, src_crs, grid, resampling)
        for i in range(field.values.shape[0])
    ]
    ts_seconds = field.timestamps.astype(float)
    return TemporalField(timestamps=list(ts_seconds), fields=reprojected)


@dataclass
class LiveEnvironmentBuilder:
    """
    Wires together all four fetchers. Each is independently optional --
    e.g. `include_wind=False` if you only care about currents -- but any
    included source that fails to fetch (missing credentials, no
    coverage, network error) raises rather than silently degrading the
    resulting EnvironmentalDataset.
    """

    grid: CommonGrid
    cache: DiskCache = None
    use_reanalysis_currents: bool = False
    include_wind: bool = True
    include_sea_ice: bool = True
    include_currents: bool = True
    include_bathymetry: bool = True

    def __post_init__(self):
        if self.cache is None:
            self.cache = DiskCache()

    def build(self, bbox: BoundingBox, time_range: TimeRange) -> EnvironmentalDataset:
        temporal_fields: dict = {}
        static_fields: dict = {}

        if self.include_wind:
            wind = ERA5WindFetcher(cache=self.cache).fetch(bbox, time_range)
            u_field = FetchedField(wind.variable_name, wind.lon, wind.lat, wind.values[0], wind.timestamps)
            v_field = FetchedField(wind.variable_name, wind.lon, wind.lat, wind.values[1], wind.timestamps)
            temporal_fields["wind_u"] = _field_to_temporal_field(u_field, "EPSG:4326", self.grid, "wind_u")
            temporal_fields["wind_v"] = _field_to_temporal_field(v_field, "EPSG:4326", self.grid, "wind_v")

        if self.include_sea_ice:
            sic = AMSRSeaIceFetcher(cache=self.cache).fetch(bbox, time_range)
            temporal_fields["sea_ice_concentration"] = _field_to_temporal_field(
                sic, "EPSG:4326", self.grid, "sea_ice_concentration"
            )

        if self.include_currents:
            currents = CopernicusMarineCurrentFetcher(
                cache=self.cache, use_reanalysis=self.use_reanalysis_currents
            ).fetch(bbox, time_range)
            u_field = FetchedField(currents.variable_name, currents.lon, currents.lat, currents.values[0], currents.timestamps)
            v_field = FetchedField(currents.variable_name, currents.lon, currents.lat, currents.values[1], currents.timestamps)
            temporal_fields["ocean_u"] = _field_to_temporal_field(u_field, "EPSG:4326", self.grid, "ocean_u")
            temporal_fields["ocean_v"] = _field_to_temporal_field(v_field, "EPSG:4326", self.grid, "ocean_v")

        if self.include_bathymetry:
            bathy = BedMachineBathymetryFetcher(cache=self.cache).fetch(bbox)
            # BedMachine is already in the working CRS (EPSG:3031) on its
            # own native x/y grid -- reproject/resample onto our grid's
            # transform directly, no lon/lat round-trip.
            src_transform = _native_xy_affine(bathy.lon, bathy.lat)  # lon/lat fields hold native x/y here
            static_fields["bathymetry"] = _reproject_2d(
                bathy.values, src_transform, self.grid.crs, self.grid, ResamplingKind.BILINEAR
            )

        latitude_field = _build_latitude_field(self.grid)

        return EnvironmentalDataset(
            transform=self.grid.transform,
            temporal_fields=temporal_fields,
            static_fields=static_fields,
            latitude_field=latitude_field,
        )


def _native_xy_affine(x: np.ndarray, y: np.ndarray):
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    west = float(x[0]) - dx / 2
    if dy > 0:
        north = float(y[-1]) + dy / 2
        row_pitch = -dy
    else:
        north = float(y[0]) - dy / 2
        row_pitch = dy
    return (dx, 0.0, west, 0.0, row_pitch, north)


def _build_latitude_field(grid: CommonGrid) -> np.ndarray:
    """Derive a latitude-degrees grid matching `grid`'s cells, for the
    Coriolis term -- computed once via pyproj, not fetched externally."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    xres, _, x0, _, yres, y0 = grid.transform
    xs = x0 + xres * (np.arange(grid.width) + 0.5)
    ys = y0 + yres * (np.arange(grid.height) + 0.5)
    xx, yy = np.meshgrid(xs, ys)
    _, lat = transformer.transform(xx, yy)
    return lat
