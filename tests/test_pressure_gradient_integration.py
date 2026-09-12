import numpy as np
import pytest

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.data.environmental_dataset import EnvironmentalDataset
from iceberg_model.data.temporal_loader import TemporalField
from iceberg_model.physics.pressure_gradient import calculate_pressure_gradient_acceleration
from iceberg_model.state.iceberg_state import IcebergState


def _sloped_ssh_dataset(grid_size=20, resolution_m=1000.0):
    shape = (grid_size, grid_size)
    transform = (resolution_m, 0.0, -grid_size * resolution_m / 2, 0.0, -resolution_m, grid_size * resolution_m / 2)
    # SSH increases linearly with x -> constant eastward gradient, no y-gradient.
    ssh = np.tile(np.linspace(0, 1.0, grid_size), (grid_size, 1))
    return EnvironmentalDataset(
        transform=transform,
        temporal_fields={"sea_surface_height": TemporalField(timestamps=[0.0, 1e6], fields=[ssh, ssh])},
    )


def test_ssh_gradient_not_computed_when_disabled():
    config = PhysicsConfig()
    assert config.pressure_gradient.enabled is False
    ds = _sloped_ssh_dataset()
    env = ds.sample_environment(x=0.0, y=0.0, t=0.0, config=config)
    assert env.ssh_gradient is None
    assert env.validity_mask["ssh_gradient"] is False


def test_ssh_gradient_computed_when_enabled_and_field_present():
    config = PhysicsConfig(pressure_gradient={"enabled": True})
    ds = _sloped_ssh_dataset()
    env = ds.sample_environment(x=0.0, y=0.0, t=0.0, config=config)
    assert env.ssh_gradient is not None
    assert env.validity_mask["ssh_gradient"] is True
    # SSH increases with x -> positive d(ssh)/dx, ~zero d(ssh)/dy
    assert env.ssh_gradient[0] > 0
    assert env.ssh_gradient[1] == pytest.approx(0.0, abs=1e-9)


def test_pressure_gradient_force_is_nonzero_end_to_end():
    """The full chain: enabled config + real SSH field -> nonzero a_pressure,
    proving the force is no longer permanently dead."""
    config = PhysicsConfig(pressure_gradient={"enabled": True})
    ds = _sloped_ssh_dataset()
    env = ds.sample_environment(x=0.0, y=0.0, t=0.0, config=config)
    state = IcebergState(x=0, y=0, u=0, v=0, L=400, W=200, H=80)

    accel = calculate_pressure_gradient_acceleration(state, env, config)
    assert not np.allclose(accel, 0.0)
    # a_pressure = -g * grad(ssh); positive x-gradient -> negative x-acceleration.
    assert accel[0] < 0


def test_pressure_gradient_missing_neighbor_returns_none_not_fabricated():
    """At the grid edge, one or more neighbor samples are out of bounds --
    the gradient must come back None/invalid, never a one-sided guess."""
    config = PhysicsConfig(pressure_gradient={"enabled": True})
    ds = _sloped_ssh_dataset(grid_size=20, resolution_m=1000.0)
    # x = -9500 is right at/near the western edge of a 20-cell, 1000m grid
    # centered at 0 -> the "west" neighbor sample falls out of bounds.
    env = ds.sample_environment(x=-9900.0, y=0.0, t=0.0, config=config)
    assert env.ssh_gradient is None
    assert env.validity_mask["ssh_gradient"] is False
