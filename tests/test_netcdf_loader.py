import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from iceberg_model.data.fetchers.base import FetchedField
from iceberg_model.data.netcdf_loader import (
    inspect_netcdf,
    netcdf_to_fetched_field,
    read_netcdf,
)


@pytest.fixture
def synthetic_netcdf(tmp_path):
    lon = np.linspace(-70, -50, 8)
    lat = np.linspace(-70, -60, 6)
    time = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")
    values = np.random.default_rng(0).normal(size=(2, 6, 8)).astype("float32")

    ds = xr.Dataset(
        {"sea_ice_concentration": (("time", "lat", "lon"), values,
                                    {"units": "fraction", "standard_name": "sea_ice_area_fraction"})},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"crs": "EPSG:4326", "title": "synthetic test dataset"},
    )
    path = tmp_path / "synthetic.nc"
    ds.to_netcdf(path)
    return path


def test_inspect_netcdf_lists_variables_without_loading_data(synthetic_netcdf):
    info = inspect_netcdf(synthetic_netcdf)
    assert "sea_ice_concentration" in info["variables"]
    assert info["variables"]["sea_ice_concentration"]["dims"] == ("time", "lat", "lon")
    assert info["global_attrs"]["title"] == "synthetic test dataset"


def test_read_netcdf_preserves_metadata(synthetic_netcdf):
    data = read_netcdf(synthetic_netcdf, "sea_ice_concentration")
    assert data.crs == "EPSG:4326"
    assert data.dims == ("time", "lat", "lon")
    assert data.values.shape == (2, 6, 8)
    assert data.variable_info.units == "fraction"
    assert data.variable_info.standard_name == "sea_ice_area_fraction"
    assert "lon" in data.coords and "lat" in data.coords and "time" in data.coords


def test_read_netcdf_unknown_variable_raises_with_available_list(synthetic_netcdf):
    with pytest.raises(KeyError, match="sea_ice_concentration"):
        read_netcdf(synthetic_netcdf, "not_a_real_variable")


def test_read_netcdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_netcdf("/nonexistent/path/file.nc", "anything")


def test_netcdf_to_fetched_field_round_trip(synthetic_netcdf):
    data = read_netcdf(synthetic_netcdf, "sea_ice_concentration")
    field = netcdf_to_fetched_field(data)
    assert isinstance(field, FetchedField)
    assert field.values.shape == (2, 6, 8)
    assert field.timestamps is not None
    assert len(field.timestamps) == 2
    assert field.units == "fraction"


def test_missing_crs_warns_but_does_not_fabricate(tmp_path):
    lon = np.linspace(-70, -50, 4)
    lat = np.linspace(-70, -60, 4)
    values = np.zeros((4, 4), dtype="float32")
    ds = xr.Dataset(
        {"bathymetry": (("lat", "lon"), values)},
        coords={"lat": lat, "lon": lon},
    )
    path = tmp_path / "no_crs.nc"
    ds.to_netcdf(path)

    with pytest.warns(UserWarning, match="no recognizable CRS"):
        data = read_netcdf(path, "bathymetry")
    assert data.crs is None
