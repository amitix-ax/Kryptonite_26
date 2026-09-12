import pytest

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.buoyancy import (
    calculate_freeboard,
    calculate_submerged_depth,
    submerged_fraction,
)
from iceberg_model.state.iceberg_state import IcebergState


def make_state(H=100.0):
    return IcebergState(x=0, y=0, u=0, v=0, L=500, W=300, H=H)


def test_submerged_depth_matches_archimedes_ratio():
    config = PhysicsConfig()
    state = make_state(H=100.0)
    D = calculate_submerged_depth(state, config)
    expected = (config.fluids.rho_ice / config.fluids.rho_seawater) * state.H
    assert D == pytest.approx(expected)


def test_freeboard_plus_submerged_equals_thickness():
    config = PhysicsConfig()
    state = make_state(H=80.0)
    D = calculate_submerged_depth(state, config)
    fb = calculate_freeboard(state, config)
    assert D + fb == pytest.approx(state.H)


def test_submerged_depth_within_thickness_bounds():
    config = PhysicsConfig()
    state = make_state(H=50.0)
    D = calculate_submerged_depth(state, config)
    assert 0 < D < state.H


def test_invalid_density_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PhysicsConfig(fluids={"rho_ice": 1100.0, "rho_seawater": 1027.0})


def test_submerged_fraction():
    config = PhysicsConfig()
    frac = submerged_fraction(config)
    assert 0 < frac < 1
