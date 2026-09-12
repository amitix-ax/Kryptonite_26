import numpy as np
import pytest

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.numerical.stochastic_solver import run_euler_maruyama
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.state.iceberg_state import DynamicsMode, EnvironmentalState, IcebergState
from iceberg_model.uncertainty.diffusion import DiffusionConfig


def test_stochastic_grounding_transition_and_noise_stops():
    """Iceberg drifts into shallow water; the solver must switch to
    GROUNDED and stop injecting velocity noise from that point on."""
    config = PhysicsConfig()
    config.melting.enabled = False
    config.coriolis_enabled = False  # isolate current-driven drift for a predictable test

    def variable_bathymetry_provider(x, y, t):
        water_depth = 500.0 if x < 200.0 else 1.0
        return EnvironmentalState(
            ocean_velocity=np.array([0.5, 0.0]),
            wind_velocity=np.array([0.0, 0.0]),
            bathymetry=water_depth,
            latitude_deg=-65.0,
            validity_mask={"ocean_velocity": True, "wind_velocity": True,
                            "bathymetry": True, "latitude_deg": True},
        )

    diffusion_cfg = DiffusionConfig(enabled=True, velocity_sigma_ms=0.05)
    initial = IcebergState(x=0, y=0, u=0.2, v=0.0, L=500, W=300, H=50)
    rng = np.random.default_rng(42)

    traj = run_euler_maruyama(initial, t_end=3600.0, dt=10.0,
                               environmental_provider=variable_bathymetry_provider,
                               config=config, diffusion_config=diffusion_cfg,
                               melt_rates_config=MeltRatesConfig(), rng=rng)

    grounding_events = [e for e in traj.events if e["event"] == "GROUNDING"]
    assert len(grounding_events) == 1  # logged exactly once, not every step
    assert DynamicsMode.GROUNDED in traj.modes
    assert traj.modes[-1] == DynamicsMode.GROUNDED


def test_stochastic_termination_on_geometry_invalid():
    config = PhysicsConfig()
    config.grounding.enabled = False
    config.melting.min_dimension_m = 40.0

    def melt_fast_provider(x, y, t):
        return EnvironmentalState(
            ocean_velocity=np.array([0.0, 0.0]),
            wind_velocity=np.array([0.0, 0.0]),
            sst=10.0,  # warm water -> fast melt
            validity_mask={"ocean_velocity": True, "wind_velocity": True, "sst": True},
        )

    initial = IcebergState(x=0, y=0, u=0.0, v=0.0, L=50, W=50, H=45)
    traj = run_euler_maruyama(initial, t_end=250 * 86400.0, dt=86400.0,
                               environmental_provider=melt_fast_provider,
                               config=config, diffusion_config=DiffusionConfig(enabled=False))

    term_events = [e for e in traj.events if e["event"] == "GEOMETRY_INVALID"]
    assert len(term_events) == 1
    assert traj.modes[-1] == DynamicsMode.TERMINATED


def test_stochastic_no_noise_when_diffusion_disabled():
    config = PhysicsConfig()
    config.grounding.enabled = False
    config.melting.enabled = False

    def calm_provider(x, y, t):
        return EnvironmentalState(ocean_velocity=np.array([0.0, 0.0]),
                                   wind_velocity=np.array([0.0, 0.0]),
                                   validity_mask={"ocean_velocity": True, "wind_velocity": True})

    initial = IcebergState(x=0, y=0, u=0.0, v=0.0, L=400, W=200, H=80)
    traj = run_euler_maruyama(initial, t_end=1000.0, dt=10.0,
                               environmental_provider=calm_provider,
                               config=config, diffusion_config=DiffusionConfig(enabled=False),
                               rng=np.random.default_rng(1))
    # With zero forcing and no diffusion, the iceberg simply doesn't move.
    assert np.allclose(traj.states[:, 0:2], 0.0)
