"""
Tests for data/live_pipeline.py's reprojection and assembly logic, using
hand-built synthetic FetchedField objects in place of real API responses
-- these exercise the actual coordinate-transform code, not the network
calls (which are monkeypatched out / never invoked).
"""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pyproj = pytest.importorskip("pyproj")

from iceberg_model.data.fetchers.base import FetchedField
from iceberg_model.data.live_pipeline import (
    _build_latitude_field,
    _field_to_temporal_field,
    _lonlat_affine,
    _reproject_2d,
)
from iceberg_model.data.reprojection import ResamplingKind, build_common_grid


def test_lonlat_affine_ascending_latitude():
    lon = np.array([-60.0, -59.0, -58.0])
    lat = np.array([-70.0, -69.0, -68.0])  # ascending
    dx, _, west, _, row_pitch, north = _lonlat_affine(lon, lat)
    assert dx == pytest.approx(1.0)
    assert row_pitch < 0  # north-up convention
    assert north > lat.max()


def test_lonlat_affine_descending_latitude():
    lon = np.array([-60.0, -59.0, -58.0])
    lat = np.array([-68.0, -69.0, -70.0])  # descending (common in NetCDF products)
    dx, _, west, _, row_pitch, north = _lonlat_affine(lon, lat)
    assert row_pitch < 0
    assert north > lat.max()


def test_reproject_2d_produces_finite_values_within_grid():
    grid = build_common_grid(crs="EPSG:3031", resolution_m=50000.0,
                              bounds=(-3000000, -3000000, 3000000, 3000000))
    lon = np.linspace(-65, -55, 20)
    lat = np.linspace(-70, -60, 20)
    values = np.full((20, 20), 5.0)
    transform = _lonlat_affine(lon, lat)

    reprojected = _reproject_2d(values, transform, "EPSG:4326", grid, ResamplingKind.BILINEAR)
    assert reprojected.shape == (grid.height, grid.width)
    assert np.isfinite(reprojected).any()  # at least some overlap with the source extent


def test_field_to_temporal_field_builds_correct_number_of_steps():
    grid = build_common_grid(crs="EPSG:3031", resolution_m=50000.0,
                              bounds=(-500000, -500000, 500000, 500000))
    lon = np.linspace(-65, -55, 10)
    lat = np.linspace(-70, -60, 10)
    values = np.stack([np.full((10, 10), 1.0), np.full((10, 10), 2.0)])  # 2 timesteps
    timestamps = np.array([0, 3600])

    field = FetchedField(variable_name="wind_u", lon=lon, lat=lat, values=values, timestamps=timestamps)
    tf = _field_to_temporal_field(field, "EPSG:4326", grid, "wind_u")

    assert len(tf.timestamps) == 2
    assert len(tf.fields) == 2
    assert tf.fields[0].shape == (grid.height, grid.width)


def test_field_to_temporal_field_requires_timestamps():
    grid = build_common_grid(crs="EPSG:3031", resolution_m=50000.0,
                              bounds=(-500000, -500000, 500000, 500000))
    lon = np.linspace(-65, -55, 5)
    lat = np.linspace(-70, -60, 5)
    field = FetchedField(variable_name="bathymetry", lon=lon, lat=lat,
                          values=np.zeros((5, 5)), timestamps=None)
    with pytest.raises(ValueError):
        _field_to_temporal_field(field, "EPSG:4326", grid, "bathymetry")


def test_build_latitude_field_matches_southern_hemisphere():
    grid = build_common_grid(crs="EPSG:3031", resolution_m=100000.0,
                              bounds=(-1000000, -1000000, 1000000, 1000000))
    lat_field = _build_latitude_field(grid)
    assert lat_field.shape == (grid.height, grid.width)
    # EPSG:3031 (Antarctic Polar Stereographic) covers only the Southern
    # Hemisphere -- every derived latitude must be negative.
    assert np.all(lat_field < 0)
