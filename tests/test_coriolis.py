import numpy as np

from iceberg_model.physics.coriolis import calculate_coriolis_acceleration, coriolis_parameter
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def test_coriolis_parameter_negative_in_southern_hemisphere():
    f = coriolis_parameter(-70.0)
    assert f < 0


def test_coriolis_parameter_zero_at_equator():
    f = coriolis_parameter(0.0)
    assert np.isclose(f, 0.0, atol=1e-12)


def test_southern_hemisphere_eastward_motion_deflects_left():
    """An eastward-moving object (u>0, v=0) in the Southern Hemisphere
    (f<0) must deflect to the LEFT of its direction of travel. Facing
    east, "left" is north (+y), so ay = -f*u must be positive since f<0.
    This matches the documented sign convention in physics/coriolis.py."""
    state = IcebergState(x=0, y=0, u=1.0, v=0.0, L=100, W=100, H=100)
    env = EnvironmentalState(latitude_deg=-65.0, validity_mask={"latitude_deg": True})
    a = calculate_coriolis_acceleration(state, env, enabled=True)
    assert a[0] == 0.0
    assert a[1] > 0


def test_coriolis_disabled_returns_zero():
    state = IcebergState(x=0, y=0, u=1.0, v=0.0, L=100, W=100, H=100)
    env = EnvironmentalState(latitude_deg=-65.0, validity_mask={"latitude_deg": True})
    a = calculate_coriolis_acceleration(state, env, enabled=False)
    assert np.allclose(a, 0.0)


def test_coriolis_missing_latitude_returns_zero():
    state = IcebergState(x=0, y=0, u=1.0, v=0.0, L=100, W=100, H=100)
    env = EnvironmentalState(latitude_deg=None)
    a = calculate_coriolis_acceleration(state, env, enabled=True)
    assert np.allclose(a, 0.0)
