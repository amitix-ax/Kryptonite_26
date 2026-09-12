import numpy as np
import pytest

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.melting import MeltRatesConfig, calculate_melt_rates
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def make_state():
    return IcebergState(x=0, y=0, u=0, v=0, L=500, W=300, H=100)


def test_no_melt_without_sst():
    config = PhysicsConfig()
    melt_cfg = MeltRatesConfig()
    state = make_state()
    env = EnvironmentalState(sst=None)
    rates = calculate_melt_rates(state, env, config, melt_cfg)
    assert rates.basal_rate == 0.0
    assert rates.lateral_rate == 0.0


def test_melt_rates_nonnegative_and_geometry_shrinks():
    config = PhysicsConfig()
    melt_cfg = MeltRatesConfig()
    state = make_state()
    env = EnvironmentalState(
        sst=2.0,
        ocean_velocity=np.array([0.3, 0.1]),
        wind_velocity=np.array([8.0, 0.0]),
        validity_mask={"sst": True},
    )
    rates = calculate_melt_rates(state, env, config, melt_cfg)
    assert rates.basal_rate >= 0
    assert rates.lateral_rate >= 0
    assert rates.wave_rate >= 0
    assert rates.dH_dt <= 0
    assert rates.dL_dt <= 0
    assert rates.dW_dt <= 0


def test_no_wave_erosion_below_threshold():
    config = PhysicsConfig()
    melt_cfg = MeltRatesConfig(wave_threshold_speed_ms=10.0)
    state = make_state()
    env = EnvironmentalState(wind_velocity=np.array([2.0, 0.0]))
    rates = calculate_melt_rates(state, env, config, melt_cfg)
    assert rates.wave_rate == 0.0


def test_disabled_melt_component_stays_zero():
    config = PhysicsConfig(melting={"basal_melt_enabled": False})
    melt_cfg = MeltRatesConfig()
    state = make_state()
    env = EnvironmentalState(sst=5.0, ocean_velocity=np.array([1.0, 0.0]),
                              validity_mask={"sst": True})
    rates = calculate_melt_rates(state, env, config, melt_cfg)
    assert rates.basal_rate == 0.0
