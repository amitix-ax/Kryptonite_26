# Data Pipeline

## Overview

```
GeoTIFF / NetCDF / xarray / numpy
        |
read_geotiff() / xarray loaders     (data/geotiff_reader.py)
        |
reproject_to_common_grid()          (data/reprojection.py)
        |
TemporalField per variable          (data/temporal_loader.py)
        |
EnvironmentalDataset.sample_environment(x, y, t)   (data/environmental_dataset.py)
        |
EnvironmentalState  ->  physics/*
```

## GeoTIFF ingestion

`data/geotiff_reader.py::read_geotiff()` returns a `GeoTiffData` object
that preserves everything: raster values, CRS, affine transform, bounds,
resolution, dtype, nodata, and file-level metadata/tags. **A TIFF is never
assumed to be RGB** — single-band scientific rasters are returned as 2D
arrays, multi-band as `(bands, rows, cols)`.

If a file has no CRS defined, a warning is raised immediately rather than
silently assuming one; downstream reprojection will refuse to guess a
source CRS.

## Reprojection to a common grid

`data/reprojection.py::reproject_to_common_grid()` moves every field onto
one shared `CommonGrid` (default CRS `EPSG:3031`, configurable resolution
and bounds). Resampling method depends on variable type:

| Variable                     | Resampling  | Why                                   |
|-------------------------------|-------------|----------------------------------------|
| `sea_ice_concentration`       | bilinear    | continuous [0,1] field; blind cubic can overshoot near ice edges |
| `ocean_u/v`, `wind_u/v`       | bilinear    | continuous vector fields |
| `sea_surface_temperature/height`, `bathymetry` | bilinear | continuous scalar fields |
| `land_mask` (categorical)     | nearest     | categorical data must not be blended |

`VARIABLE_RESAMPLING_DEFAULTS` in `reprojection.py` is the single source
of truth for this mapping; cubic is never applied blindly.

## A TIFF is a spatial field, not a trajectory

A raster represents `E(x, y, t)` — e.g. `SIC(x, y, t)` or `U_ocean(x, y, t)`.
Turning that into forcing for a specific iceberg requires sampling it at
the iceberg's current position and time — that's the entire job of
`environmental_dataset.py::sample_environment()`.

## Temporal interpolation

`data/temporal_loader.py::interpolate_in_time()` linearly interpolates
between the two bracketing timestamps:

```
alpha = (t - t0) / (t1 - t0)
E(t) = (1 - alpha) * E(t0) + alpha * E(t1)
```

Extrapolation outside the dataset's time range raises `TimeOutOfBoundsError`
**unless** `PhysicsConfig.temporal_interpolation.allow_extrapolation` is
explicitly set — environmental forcing is never silently extrapolated.

## Missing data

Every sampled field comes back through `EnvironmentalState.validity_mask`.
A `None` field with `validity_mask[name] = False` means "this force term
is skipped this step," never "assume zero/typical." This is enforced
throughout `physics/*` (e.g. `ocean_drag.py` returns a zero acceleration,
not a fabricated one, when `ocean_velocity` is unavailable).

## Supported formats

- **GeoTIFF** — primary format, via `rasterio` (`data/geotiff_reader.py`).
- **NetCDF / xarray** — `EnvironmentalDataset` is format-agnostic once
  data is reprojected onto the common grid; `xarray`-based loaders can
  populate the same `TemporalField`/static-field structures used by the
  GeoTIFF path. (Loader stubs live alongside `temporal_loader.py`; wire in
  an xarray-specific reader following the same "preserve all metadata,
  reject silent guesses" pattern as `geotiff_reader.py` when NetCDF inputs
  are available.)
- **numpy arrays** — any field can be supplied directly as a `TemporalField`/
  static array, as shown in `examples/synthetic_environment.py`.

## Automatic live data ingestion (`data/fetchers/`, `data/live_pipeline.py`)

Rather than manually downloading GeoTIFF/NetCDF files, `LiveEnvironmentBuilder`
(`data/live_pipeline.py`) pulls all four environmental inputs directly
from their source APIs and assembles a ready-to-use `EnvironmentalDataset`:

| Field                     | Source                          | Client library     |
|----------------------------|----------------------------------|----------------------|
| Wind (u10, v10)            | ERA5 reanalysis (Copernicus CDS) | `cdsapi`             |
| Sea-ice concentration      | AMSR2/GCOM-W1 AU_SI25 (NSIDC)    | `earthaccess`        |
| Ocean currents (uo, vo)    | Copernicus Marine Service        | `copernicusmarine`   |
| Bathymetry                 | BedMachine Antarctica (NSIDC)    | `earthaccess`        |

Usage:

```python
from iceberg_model.data.fetchers.common import BoundingBox, TimeRange
from iceberg_model.data.live_pipeline import LiveEnvironmentBuilder
from iceberg_model.data.reprojection import build_common_grid

grid = build_common_grid(crs="EPSG:3031", resolution_m=5000.0,
                          bounds=(-2_000_000, -2_000_000, 2_000_000, 2_000_000))
bbox = BoundingBox(min_lon=-60, min_lat=-66, max_lon=-55, max_lat=-63)
time_range = TimeRange(start=..., end=...)

environment = LiveEnvironmentBuilder(grid=grid).build(bbox, time_range)
```

See `examples/fetch_and_run_live.py` for a full runnable script (CLI
args for bbox/date/iceberg geometry) and `pyproject.toml`'s `live-data`
extra for the required packages.

### Setup (all four services are free)

1. `pip install -e ".[live-data]"`
2. Copy `.env.example` to `.env` and fill in credentials:
   - **CDS** (ERA5 wind): register at https://cds.climate.copernicus.eu/,
     set `CDSAPI_KEY`.
   - **NASA Earthdata** (AMSR2 sea ice, BedMachine bathymetry — both
     NSIDC-hosted): register at https://urs.earthdata.nasa.gov/, set
     `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD` (or `EARTHDATA_TOKEN`).
   - **Copernicus Marine** (ocean currents): register at
     https://data.marine.copernicus.eu/register, set
     `COPERNICUSMARINE_SERVICE_USERNAME` + `COPERNICUSMARINE_SERVICE_PASSWORD`.
3. `.env` is loaded automatically via `credentials.load_dotenv_if_present()`
   at the top of `examples/fetch_and_run_live.py`, or export the same
   variables directly in your shell/CI environment.

### Design choices worth knowing

- **Caching** (`fetchers/common.py::DiskCache`): every fetch is
  content-addressed by (source, parameters) and cached under
  `data/cache/`. Time-varying products default to a 7-day freshness
  window; BedMachine (a static product) is cached indefinitely.
- **Never silently degrades**: a fetcher with missing credentials, no
  coverage for the requested bbox/time, or an unrecognized variable
  schema raises a specific, actionable exception rather than returning a
  partial or fabricated `EnvironmentalDataset`. `LiveEnvironmentBuilder`
  lets you opt individual sources out (`include_wind=False`, etc.) but
  never fails open on an included one.
- **Reprojection is uniform**: every fetcher returns geographic (lon/lat)
  data except BedMachine (native EPSG:3031); `live_pipeline.py` reprojects
  all of them onto the same `CommonGrid` using the same resampling rules
  as `reprojection.py` (bilinear for continuous fields, nearest for
  categorical).
- **Sandbox limitation**: the environment that built this repository has
  network egress restricted to package registries and cannot reach any
  of the four services above. The fetcher code calls each service's real
  client library with correct parameters and is validated by mocked/
  synthetic-data tests (`tests/test_live_data_fetchers.py`,
  `tests/test_live_pipeline.py`); it has not been (and could not be)
  exercised against the live endpoints from this environment. Run
  `examples/fetch_and_run_live.py` from a machine with open network
  access to validate against real data.
