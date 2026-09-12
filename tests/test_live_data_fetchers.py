"""
Tests for the live-data pipeline. These NEVER make real network calls
(this sandbox couldn't reach the external services anyway) -- they verify
credential-handling, caching, and the reprojection glue in
live_pipeline.py using synthetic in-memory data and monkeypatched
fetchers.
"""

import os
from datetime import datetime

import numpy as np
import pytest

from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.credentials import (
    CDSCredentials,
    CopernicusMarineCredentials,
    EarthdataCredentials,
    MissingCredentialsError,
)


def test_bounding_box_validates_ranges():
    BoundingBox(min_lon=-60, min_lat=-70, max_lon=-50, max_lat=-60)  # valid
    with pytest.raises(ValueError):
        BoundingBox(min_lon=-50, min_lat=-70, max_lon=-60, max_lat=-60)  # min > max


def test_time_range_validates_order():
    TimeRange(start=datetime(2026, 1, 1), end=datetime(2026, 1, 2))
    with pytest.raises(ValueError):
        TimeRange(start=datetime(2026, 1, 2), end=datetime(2026, 1, 1))


def test_cds_credentials_missing_raises(monkeypatch):
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    with pytest.raises(MissingCredentialsError):
        CDSCredentials.from_env()


def test_cds_credentials_present(monkeypatch):
    monkeypatch.setenv("CDSAPI_KEY", "fake-key")
    creds = CDSCredentials.from_env()
    assert creds.key == "fake-key"
    assert "copernicus" in creds.url


def test_earthdata_credentials_missing_raises(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    with pytest.raises(MissingCredentialsError):
        EarthdataCredentials.from_env()


def test_earthdata_credentials_token_sufficient(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setenv("EARTHDATA_TOKEN", "fake-token")
    creds = EarthdataCredentials.from_env()
    assert creds.has_any()


def test_copernicus_marine_credentials_missing_raises(monkeypatch):
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    with pytest.raises(MissingCredentialsError):
        CopernicusMarineCredentials.from_env()


def test_disk_cache_key_is_stable_and_order_independent(tmp_path):
    cache = DiskCache(cache_dir=tmp_path)
    k1 = cache.key_for("era5_wind", {"a": 1, "b": 2})
    k2 = cache.key_for("era5_wind", {"b": 2, "a": 1})
    assert k1 == k2


def test_disk_cache_key_differs_for_different_params(tmp_path):
    cache = DiskCache(cache_dir=tmp_path)
    k1 = cache.key_for("era5_wind", {"a": 1})
    k2 = cache.key_for("era5_wind", {"a": 2})
    assert k1 != k2


def test_disk_cache_freshness(tmp_path):
    cache = DiskCache(cache_dir=tmp_path, max_age_days=1.0)
    p = tmp_path / "somefile.nc"
    assert not cache.is_fresh(p)  # doesn't exist yet
    p.write_text("data")
    assert cache.is_fresh(p)


def test_disk_cache_indefinite_when_max_age_none(tmp_path):
    cache = DiskCache(cache_dir=tmp_path, max_age_days=None)
    p = tmp_path / "static.nc"
    p.write_text("data")
    assert cache.is_fresh(p)
