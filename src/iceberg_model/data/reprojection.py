"""
Reprojection to a common working grid (spec section 8).

Pipeline: source CRS -> target CRS -> target resolution -> target bounds ->
resampled raster.

Resampling method matters: continuous variables (ocean current, SST, SSH,
bathymetry) get bilinear/cubic; categorical masks get nearest-neighbor;
sea-ice concentration gets a continuous-aware method (bilinear by default,
NOT blind cubic, which can overshoot outside [0,1] near ice edges).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from iceberg_model.data.geotiff_reader import GeoTiffData


class ResamplingKind(str, Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"


VARIABLE_RESAMPLING_DEFAULTS: dict[str, ResamplingKind] = {
    "sea_ice_concentration": ResamplingKind.BILINEAR,  # continuous-aware; never blind cubic
    "ocean_u": ResamplingKind.BILINEAR,
    "ocean_v": ResamplingKind.BILINEAR,
    "wind_u": ResamplingKind.BILINEAR,
    "wind_v": ResamplingKind.BILINEAR,
    "sea_surface_temperature": ResamplingKind.BILINEAR,
    "sea_surface_height": ResamplingKind.BILINEAR,
    "bathymetry": ResamplingKind.BILINEAR,
    "land_mask": ResamplingKind.NEAREST,  # categorical
}


@dataclass
class CommonGrid:
    crs: str
    resolution_m: float
    bounds: tuple  # (left, bottom, right, top) in `crs`
    width: int
    height: int
    transform: tuple


def build_common_grid(crs: str, resolution_m: float, bounds: tuple) -> CommonGrid:
    left, bottom, right, top = bounds
    width = int(np.ceil((right - left) / resolution_m))
    height = int(np.ceil((top - bottom) / resolution_m))
    transform = (resolution_m, 0.0, left, 0.0, -resolution_m, top)
    return CommonGrid(crs=crs, resolution_m=resolution_m, bounds=bounds,
                       width=width, height=height, transform=transform)


def reproject_to_common_grid(data: GeoTiffData, grid: CommonGrid,
                              variable_name: Optional[str] = None,
                              resampling: Optional[ResamplingKind] = None) -> np.ndarray:
    """
    Reproject `data.raster_values` from its source CRS/transform onto
    `grid`. Requires rasterio. Resampling method is chosen from
    (in priority order): explicit `resampling` arg, then
    VARIABLE_RESAMPLING_DEFAULTS[variable_name], then bilinear.
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import Affine
    except ImportError as e:
        raise ImportError(
            "rasterio is required for reprojection. Install with `pip install rasterio`."
        ) from e

    if data.crs is None:
        raise ValueError(
            "Cannot reproject: source GeoTIFF has no CRS. Supply an explicit "
            "source CRS rather than guessing (principle J: missing metadata "
            "must never silently become a valid assumption)."
        )

    if resampling is None:
        resampling = VARIABLE_RESAMPLING_DEFAULTS.get(variable_name, ResamplingKind.BILINEAR)

    resampling_map = {
        ResamplingKind.NEAREST: Resampling.nearest,
        ResamplingKind.BILINEAR: Resampling.bilinear,
        ResamplingKind.CUBIC: Resampling.cubic,
    }

    src_transform = Affine(*data.transform)
    dst_transform = Affine(*grid.transform)

    src = data.raster_values
    if src.ndim == 2:
        src = src[np.newaxis, ...]

    dst = np.full((src.shape[0], grid.height, grid.width), np.nan, dtype=np.float64)

    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=data.crs,
        dst_transform=dst_transform,
        dst_crs=grid.crs,
        src_nodata=data.nodata,
        dst_nodata=np.nan,
        resampling=resampling_map[resampling],
    )

    return dst[0] if dst.shape[0] == 1 else dst
