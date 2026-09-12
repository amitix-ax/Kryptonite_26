import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.data.environmental_dataset import EnvironmentalDataset
from iceberg_model.data.temporal_loader import TemporalField
from iceberg_model.io.trajectory_export import export_csv
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.simulation.simulator import Simulator
from iceberg_model.state.iceberg_state import DynamicsMode, IcebergState


def build_synthetic_environment(grid_size=50, resolution_m=2000.0):
    """
    Build a small synthetic EnvironmentalDataset entirely in memory:
    - A uniform eastward ocean current
    - A uniform southerly wind
    - Sea-ice concentration decaying with x
    - Deep bathymetry everywhere except a shallow patch far from the start
    - A latitude field fixed at -65 degrees (Southern Ocean)
    """
    shape = (grid_size, grid_size)
    transform = (resolution_m, 0.0, -grid_size * resolution_m / 2, 0.0, -resolution_m, grid_size * resolution_m / 2)

    ocean_u = np.full(shape, 0.3)
    ocean_v = np.full(shape, 0.0)
    wind_u = np.full(shape, 0.0)
    wind_v = np.full(shape, -4.0)
    sic = np.clip(1.0 - np.linspace(0, 1, grid_size)[None, :].repeat(grid_size, axis=0), 0, 1)
    sst = np.full(shape, 1.5)
    bathymetry = np.full(shape, 500.0)
    latitude = np.full(shape, -65.0)

    ds = EnvironmentalDataset(
        transform=transform,
        temporal_fields={
            "ocean_u": TemporalField(timestamps=[0.0, 1e6], fields=[ocean_u, ocean_u]),
            "ocean_v": TemporalField(timestamps=[0.0, 1e6], fields=[ocean_v, ocean_v]),
            "wind_u": TemporalField(timestamps=[0.0, 1e6], fields=[wind_u, wind_u]),
            "wind_v": TemporalField(timestamps=[0.0, 1e6], fields=[wind_v, wind_v]),
            "sea_ice_concentration": TemporalField(timestamps=[0.0, 1e6], fields=[sic, sic]),
            "sea_surface_temperature": TemporalField(timestamps=[0.0, 1e6], fields=[sst, sst]),
        },
        static_fields={"bathymetry": bathymetry},
        latitude_field=latitude,
    )
    return ds


def test_end_to_end_synthetic_simulation_runs_and_exports(tmp_path):
    config = PhysicsConfig()
    env = build_synthetic_environment()

    simulator = Simulator(config=config, environment=env, melt_rates_config=MeltRatesConfig())

    initial_state = IcebergState(x=0.0, y=0.0, u=0.0, v=0.0, L=400.0, W=250.0, H=80.0)
    result = simulator.run(initial_state, t_end=6 * 3600.0)  # 6 hours

    times, states = result.concatenated()

    assert len(times) > 1
    assert np.all(np.isfinite(states))

    # Iceberg should have moved (nonzero displacement) under forcing.
    displacement = np.linalg.norm(states[-1, 0:2] - states[0, 0:2])
    assert displacement > 0

    # Geometry should not have grown.
    assert states[-1, 4] <= states[0, 4]  # L
    assert states[-1, 5] <= states[0, 5]  # W
    assert states[-1, 6] <= states[0, 6]  # H

    out_path = tmp_path / "trajectory.csv"
    export_csv(result, out_path)
    assert out_path.exists()
    content = out_path.read_text()
    assert "time_s" in content.splitlines()[0]
    assert len(content.splitlines()) > 1


def test_end_to_end_residual_model_none_is_supported():
    """The physics engine must run correctly with ResidualModel=None
    (spec section 25) -- this is the default and only configuration
    exercised in this task."""
    config = PhysicsConfig()
    env = build_synthetic_environment()
    simulator = Simulator(config=config, environment=env, residual_model=None)
    initial_state = IcebergState(x=0.0, y=0.0, u=0.0, v=0.0, L=300.0, W=200.0, H=60.0)
    result = simulator.run(initial_state, t_end=3600.0)
    assert result.final_mode in (DynamicsMode.FREE_FLOATING, DynamicsMode.GROUNDED, DynamicsMode.TERMINATED)
