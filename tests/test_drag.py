import numpy as np
import pytest

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.ocean_drag import calculate_ocean_drag_acceleration
from iceberg_model.physics.wind_drag import calculate_wind_drag_acceleration
from iceberg_model.physics.forces import iceberg_mass
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def make_state():
    return IcebergState(x=0, y=0, u=0.0, v=0.0, L=500, W=300, H=100)


def test_ocean_drag_zero_when_no_relative_velocity():
    config = PhysicsConfig()
    state = IcebergState(x=0, y=0, u=1.0, v=0.5, L=500, W=300, H=100)
    env = EnvironmentalState(ocean_velocity=np.array([1.0, 0.5]), validity_mask={"ocean_velocity": True})
    mass = iceberg_mass(state, config)
    result = calculate_ocean_drag_acceleration(state, env, config, mass)
    assert np.allclose(result.acceleration, 0.0, atol=1e-10)


def test_ocean_drag_points_toward_relative_current():
    config = PhysicsConfig()
    state = make_state()
    env = EnvironmentalState(ocean_velocity=np.array([0.5, 0.0]), validity_mask={"ocean_velocity": True})
    mass = iceberg_mass(state, config)
    result = calculate_ocean_drag_acceleration(state, env, config, mass)
    assert result.acceleration[0] > 0
    assert result.acceleration[1] == pytest.approx(0.0, abs=1e-12)


def test_ocean_drag_missing_data_returns_zero():
    config = PhysicsConfig()
    state = make_state()
    env = EnvironmentalState(ocean_velocity=None)
    mass = iceberg_mass(state, config)
    result = calculate_ocean_drag_acceleration(state, env, config, mass)
    assert np.allclose(result.acceleration, 0.0)


def test_wind_drag_missing_data_returns_zero():
    config = PhysicsConfig()
    state = make_state()
    env = EnvironmentalState(wind_velocity=None)
    mass = iceberg_mass(state, config)
    accel = calculate_wind_drag_acceleration(state, env, config, mass)
    assert np.allclose(accel, 0.0)


def test_wind_drag_direction():
    config = PhysicsConfig()
    state = make_state()
    env = EnvironmentalState(wind_velocity=np.array([0.0, 10.0]), validity_mask={"wind_velocity": True})
    mass = iceberg_mass(state, config)
    accel = calculate_wind_drag_acceleration(state, env, config, mass)
    assert accel[1] > 0
    assert accel[0] == pytest.approx(0.0, abs=1e-12)
