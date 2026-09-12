"""
GeoTIFF ingestion (spec section 8).

A TIFF is never assumed to be a simple RGB image -- it is treated as one or
more spatial raster bands with an associated CRS, affine transform, bounds,
resolution, nodata value, and metadata, all of which are preserved and
returned rather than discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class GeoTiffData:
    """Everything read_geotiff() returns. Nothing here is discarded."""

    raster_values: np.ndarray          # shape (bands, rows, cols) or (rows, cols)
    crs: Optional[str]                  # e.g. "EPSG:3031"; None if undefined in file
    transform: tuple                    # affine transform (a, b, c, d, e, f)
    bounds: tuple                        # (left, bottom, right, top)
    width: int
    height: int
    resolution: tuple                    # (x_res, y_res)
    dtype: str
    nodata: Optional[float]
    band_count: int
    metadata: dict = field(default_factory=dict)
    band_descriptions: list = field(default_factory=list)
    source_path: str = ""


def read_geotiff(path: str | Path) -> GeoTiffData:
    """
    Read a GeoTIFF and return a GeoTiffData with full geospatial metadata
    intact. Requires `rasterio`. Raises informative errors rather than
    silently guessing at missing CRS/nodata.
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "rasterio is required to read GeoTIFF files. Install with "
            "`pip install rasterio`."
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        values = src.read()  # (bands, rows, cols)
        crs = src.crs.to_string() if src.crs else None
        transform = tuple(src.transform)[:6]
        bounds = tuple(src.bounds)
        resolution = src.res
        nodata = src.nodata
        metadata = dict(src.tags())
        band_descriptions = list(src.descriptions) if src.descriptions else []

        if values.shape[0] == 1:
            raster_values = values[0]
        else:
            raster_values = values

    if crs is None:
        # Do not silently assume a CRS -- surface this loudly so downstream
        # reprojection logic (reprojection.py) can raise instead of guessing.
        import warnings
        warnings.warn(
            f"GeoTIFF at {path} has no CRS defined. Downstream reprojection "
            "will fail unless a CRS is explicitly supplied."
        )

    return GeoTiffData(
        raster_values=raster_values,
        crs=crs,
        transform=transform,
        bounds=bounds,
        width=values.shape[-1],
        height=values.shape[-2],
        resolution=resolution,
        dtype=str(values.dtype),
        nodata=nodata,
        band_count=values.shape[0],
        metadata=metadata,
        band_descriptions=band_descriptions,
        source_path=str(path),
    )


def apply_nodata_mask(data: GeoTiffData) -> np.ma.MaskedArray:
    """Return a masked array with nodata values masked out. Never silently
    treats nodata as a valid physical value (principle J)."""
    if data.nodata is None:
        return np.ma.masked_invalid(data.raster_values)
    return np.ma.masked_equal(data.raster_values, data.nodata)
