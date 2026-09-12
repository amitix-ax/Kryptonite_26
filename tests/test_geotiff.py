import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from iceberg_model.data.geotiff_reader import apply_nodata_mask, read_geotiff


@pytest.fixture
def synthetic_geotiff(tmp_path):
    from rasterio.transform import from_origin

    path = tmp_path / "synthetic.tif"
    data = np.arange(100, dtype="float32").reshape(10, 10)
    data[0, 0] = -9999.0  # nodata sentinel

    transform = from_origin(0, 100, 10, 10)  # 10 m pixels, origin (0, 100)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="float32", crs="EPSG:3031", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    return path


def test_read_geotiff_preserves_metadata(synthetic_geotiff):
    result = read_geotiff(synthetic_geotiff)
    assert result.crs == "EPSG:3031"
    assert result.width == 10
    assert result.height == 10
    assert result.nodata == -9999.0
    assert result.band_count == 1
    assert result.raster_values.shape == (10, 10)


def test_geotiff_not_assumed_rgb(synthetic_geotiff):
    result = read_geotiff(synthetic_geotiff)
    # Single-band scientific raster, not a 3-channel RGB image.
    assert result.raster_values.ndim == 2


def test_apply_nodata_mask_masks_sentinel(synthetic_geotiff):
    result = read_geotiff(synthetic_geotiff)
    masked = apply_nodata_mask(result)
    assert masked.mask[0, 0] == True  # noqa: E712


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_geotiff("/nonexistent/path/does_not_exist.tif")
