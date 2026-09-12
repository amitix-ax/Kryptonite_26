"""
NetCDF / xarray ingestion -- the counterpart to geotiff_reader.py for the
other primary format named in the spec (section 7-8: "The data pipeline
must support GeoTIFF, NetCDF, xarray Dataset, numpy arrays").

Same guarantees as geotiff_reader.py: nothing is discarded. CRS, all
coordinate variables, variable-level and global attributes (units,
_FillValue, standard_name, etc.) are preserved and returned rather than
silently dropped. A NetCDF file is never assumed to already be on a
regular lon/lat grid with a specific variable-naming convention -- this
module inspects and reports what's actually there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class NetCDFVariableInfo:
    name: str
    dims: tuple
    shape: tuple
    dtype: str
    units: str = ""
    standard_name: str = ""
    fill_value: Optional[float] = None
    attrs: dict = field(default_factory=dict)


@dataclass
class NetCDFData:
    """Everything read_netcdf() returns for one requested variable."""

    variable_name: str
    values: np.ndarray                 # shape matches `dims`
    dims: tuple                          # e.g. ("time", "lat", "lon")
    coords: dict                          # dim name -> coordinate array
    crs: Optional[str]                    # from a recognized grid-mapping attr, or None
    global_attrs: dict = field(default_factory=dict)
    variable_info: Optional[NetCDFVariableInfo] = None
    source_path: str = ""


def inspect_netcdf(path: str | Path) -> dict:
    """
    Return a dict describing every variable/coordinate/attribute in the
    file WITHOUT loading data arrays into memory -- the NetCDF analogue of
    "never assume a TIFF is simply an RGB image": inspect before assuming
    a schema.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError("xarray is required to read NetCDF files. Install with `pip install xarray netCDF4`.") from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    with xr.open_dataset(path) as ds:
        return {
            "variables": {
                name: {
                    "dims": da.dims,
                    "shape": da.shape,
                    "dtype": str(da.dtype),
                    "attrs": dict(da.attrs),
                }
                for name, da in ds.data_vars.items()
            },
            "coordinates": list(ds.coords.keys()),
            "global_attrs": dict(ds.attrs),
        }


def _detect_crs(ds) -> Optional[str]:
    """
    Look for a recognized CRS hint: a `grid_mapping` variable (CF
    convention) or a dataset-level `crs`/`spatial_ref` attribute. Returns
    None (never a guessed default) if nothing is found -- matching
    geotiff_reader.py's behavior of warning rather than assuming EPSG:4326.
    """
    for attr_name in ("crs", "spatial_ref", "proj4", "esri_pe_string"):
        if attr_name in ds.attrs:
            return str(ds.attrs[attr_name])

    for var in ds.data_vars.values():
        gm = var.attrs.get("grid_mapping")
        if gm and gm in ds.variables:
            gm_var = ds[gm]
            if "crs_wkt" in gm_var.attrs:
                return str(gm_var.attrs["crs_wkt"])
            if "spatial_ref" in gm_var.attrs:
                return str(gm_var.attrs["spatial_ref"])

    return None


def read_netcdf(path: str | Path, variable_name: str) -> NetCDFData:
    """
    Read a single variable from a NetCDF file with its coordinates and
    full attribute metadata intact. Requires `xarray` (+ `netCDF4` or
    `h5netcdf` as the backend engine).
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError("xarray is required to read NetCDF files. Install with `pip install xarray netCDF4`.") from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    with xr.open_dataset(path) as ds:
        if variable_name not in ds.data_vars:
            available = list(ds.data_vars.keys())
            raise KeyError(
                f"Variable '{variable_name}' not found in {path}. "
                f"Available variables: {available}. Refusing to guess a "
                "substitute -- call inspect_netcdf() first if unsure."
            )

        da = ds[variable_name]
        values = da.values
        dims = da.dims
        coords = {dim: ds[dim].values for dim in dims if dim in ds.coords}
        crs = _detect_crs(ds)

        if crs is None:
            import warnings
            warnings.warn(
                f"NetCDF variable '{variable_name}' in {path} has no "
                "recognizable CRS metadata (no grid_mapping / crs / "
                "spatial_ref attribute found). Downstream reprojection "
                "will require an explicitly supplied source CRS."
            )

        info = NetCDFVariableInfo(
            name=variable_name,
            dims=dims,
            shape=values.shape,
            dtype=str(values.dtype),
            units=str(da.attrs.get("units", "")),
            standard_name=str(da.attrs.get("standard_name", "")),
            fill_value=da.attrs.get("_FillValue"),
            attrs=dict(da.attrs),
        )

    return NetCDFData(
        variable_name=variable_name,
        values=values,
        dims=dims,
        coords=coords,
        crs=crs,
        global_attrs=dict(ds.attrs) if hasattr(ds, "attrs") else {},
        variable_info=info,
        source_path=str(path),
    )


def netcdf_to_fetched_field(data: NetCDFData, lon_dim: str = "lon", lat_dim: str = "lat",
                             time_dim: str = "time"):
    """
    Convert a NetCDFData (from a regular lon/lat/time grid) into the same
    `FetchedField` shape used by the live-data fetchers
    (data/fetchers/base.py), so a locally downloaded NetCDF file can flow
    through the exact same reprojection/assembly code path as a live
    fetch (data/live_pipeline.py) -- one code path, not two.
    """
    from iceberg_model.data.fetchers.base import FetchedField

    if lon_dim not in data.coords or lat_dim not in data.coords:
        raise KeyError(
            f"Expected coordinate dims '{lon_dim}'/'{lat_dim}' not found "
            f"in {data.dims}. Pass the correct lon_dim/lat_dim for this "
            "file's actual coordinate naming."
        )

    lon = data.coords[lon_dim]
    lat = data.coords[lat_dim]
    values = data.values

    timestamps = None
    if time_dim in data.coords:
        timestamps = data.coords[time_dim].astype("datetime64[s]").astype("int64")
    elif time_dim not in data.dims:
        # Static field (e.g. bathymetry with no time dimension) -- leave
        # timestamps as None, matching FetchedField's static-field convention.
        pass

    return FetchedField(
        variable_name=data.variable_name,
        lon=lon, lat=lat,
        values=values,
        timestamps=timestamps,
        units=data.variable_info.units if data.variable_info else "",
        source=f"NetCDF file: {data.source_path}",
    )
