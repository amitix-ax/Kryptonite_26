import numpy as np
import pytest

from iceberg_model.data.interpolation import bilinear_interpolate
from iceberg_model.data.temporal_loader import TemporalField, TimeOutOfBoundsError, interpolate_in_time


def test_bilinear_interpolate_exact_grid_point():
    field = np.array([[0.0, 1.0], [2.0, 3.0]])
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 1.0)  # 1m pixels, origin (0,1), north-up
    result = bilinear_interpolate(field, transform, x=0.0, y=1.0)
    assert result.valid
    assert result.value == pytest.approx(0.0)


def test_bilinear_interpolate_midpoint():
    field = np.array([[0.0, 10.0], [0.0, 10.0]])
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
    result = bilinear_interpolate(field, transform, x=0.5, y=0.5)
    assert result.valid
    assert result.value == pytest.approx(5.0)


def test_bilinear_interpolate_out_of_bounds():
    field = np.zeros((5, 5))
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
    result = bilinear_interpolate(field, transform, x=100.0, y=100.0)
    assert not result.valid
    assert result.reason == "out_of_bounds"


def test_bilinear_interpolate_nodata_nan():
    field = np.array([[np.nan, 1.0], [2.0, 3.0]])
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
    result = bilinear_interpolate(field, transform, x=0.0, y=1.0)
    assert not result.valid
    assert result.reason == "nodata"


def test_temporal_interpolation_linear():
    tf = TemporalField(timestamps=[0.0, 10.0], fields=[np.zeros((2, 2)), np.ones((2, 2)) * 10])
    result = interpolate_in_time(tf, 5.0)
    assert np.allclose(result, 5.0)


def test_temporal_interpolation_exact_timestamp():
    tf = TemporalField(timestamps=[0.0, 10.0], fields=[np.zeros((2, 2)), np.ones((2, 2)) * 10])
    result = interpolate_in_time(tf, 10.0)
    assert np.allclose(result, 10.0)


def test_temporal_extrapolation_rejected_by_default():
    tf = TemporalField(timestamps=[0.0, 10.0], fields=[np.zeros((2, 2)), np.ones((2, 2))])
    with pytest.raises(TimeOutOfBoundsError):
        interpolate_in_time(tf, 20.0)


def test_temporal_extrapolation_allowed_when_enabled():
    tf = TemporalField(timestamps=[0.0, 10.0], fields=[np.zeros((2, 2)), np.ones((2, 2)) * 5])
    result = interpolate_in_time(tf, 20.0, allow_extrapolation=True)
    assert np.allclose(result, 5.0)
