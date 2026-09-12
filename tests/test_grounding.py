import numpy as np
import pytest

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.buoyancy import calculate_submerged_depth
from iceberg_model.physics.forces import iceberg_mass
from iceberg_model.physics.grounding import calculate_grounding_acceleration, is_grounded, normal_force
from iceberg_model.state.iceberg_state import IcebergState


def test_is_grounded_true_when_keel_exceeds_water_depth():
    config = PhysicsConfig()
    state = IcebergState(x=0, y=0, u=0, v=0, L=500, W=300, H=200)
    D = calculate_submerged_depth(state, config)
    assert is_grounded(state, config, water_depth_m=D - 1.0)


def test_is_grounded_false_when_water_deep_enough():
    config = PhysicsConfig()
    state = IcebergState(x=0, y=0, u=0, v=0, L=500, W=300, H=50)
    D = calculate_submerged_depth(state, config)
    assert not is_grounded(state, config, water_depth_m=D + 500.0)


def test_grounding_friction_opposes_velocity():
    config = PhysicsConfig()
    state = IcebergState(x=0, y=0, u=2.0, v=0.0, L=500, W=300, H=100)
    mass = iceberg_mass(state, config)
    accel = calculate_grounding_acceleration(state, config, mass, grounded=True)
    assert accel[0] < 0  # friction opposes +u motion
    assert accel[1] == pytest.approx(0.0, abs=1e-9)


def test_grounding_acceleration_zero_when_not_grounded():
    config = PhysicsConfig()
    state = IcebergState(x=0, y=0, u=2.0, v=0.0, L=500, W=300, H=100)
    mass = iceberg_mass(state, config)
    accel = calculate_grounding_acceleration(state, config, mass, grounded=False)
    assert np.allclose(accel, 0.0)


def test_normal_force_non_negative():
    config = PhysicsConfig()
    state = IcebergState(x=0, y=0, u=0, v=0, L=500, W=300, H=100)
    mass = iceberg_mass(state, config)
    N = normal_force(state, config, mass)
    assert N >= 0
