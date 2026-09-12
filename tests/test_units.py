import numpy as np
import pytest

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.forces import iceberg_mass
from iceberg_model.physics.melting import SECONDS_PER_DAY, MeltRatesConfig, calculate_melt_rates
from iceberg_model.physics.ocean_drag import calculate_ocean_drag_acceleration
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def test_iceberg_mass_is_kilograms_and_positive():
    config = PhysicsConfig()
    state = IcebergState(x=0, y=0, u=0, v=0, L=500, W=300, H=100)
    mass = iceberg_mass(state, config)
    # V = 500*300*100 = 1.5e7 m^3, rho_ice=900 kg/m^3 => 1.35e10 kg
    assert mass == pytest.approx(config.fluids.rho_ice * 500 * 300 * 100)
    assert mass > 0


def test_ocean_drag_acceleration_order_of_magnitude_is_ms2():
    """A 0.5 m/s current-iceberg speed difference should produce an
    acceleration on the order of 1e-6 to 1e-2 m/s^2, not something
    absurd like 1e5 (a unit-conversion bug) or 1e-20 (a units-mismatch
    silently zeroing the force)."""
    config = PhysicsConfig()
    state = IcebergState(x=0, y=0, u=0.0, v=0.0, L=500, W=300, H=100)
    env = EnvironmentalState(ocean_velocity=np.array([0.5, 0.0]), validity_mask={"ocean_velocity": True})
    mass = iceberg_mass(state, config)
    result = calculate_ocean_drag_acceleration(state, env, config, mass)
    a = np.linalg.norm(result.acceleration)
    assert 1e-8 < a < 1e-1


def test_melt_rate_conversion_m_per_day_to_m_per_s():
    assert SECONDS_PER_DAY == 86400.0
    config = PhysicsConfig()
    melt_cfg = MeltRatesConfig(basal_A=1.0, basal_B=0.0, lateral_k=0.0, wave_k=0.0)
    state = IcebergState(x=0, y=0, u=0, v=0, L=500, W=300, H=100)
    env = EnvironmentalState(sst=1.0, ocean_velocity=np.array([0.0, 0.0]),
                              validity_mask={"sst": True})
    rates = calculate_melt_rates(state, env, config, melt_cfg)
    # basal_A=1.0 m/day intercept -> should equal 1.0/86400 m/s
    assert rates.basal_rate == pytest.approx(1.0 / SECONDS_PER_DAY)


def test_state_vector_round_trip_preserves_units():
    state = IcebergState(x=1000.0, y=-500.0, u=0.3, v=-0.1, L=400.0, W=250.0, H=80.0)
    vec = state.as_vector()
    restored = IcebergState.from_vector(vec)
    assert restored.x == state.x
    assert restored.u == state.u
    assert restored.H == state.H
