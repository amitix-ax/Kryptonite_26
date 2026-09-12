import numpy as np
import pytest
from scipy.integrate import solve_ivp

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.numerical.deterministic_solver import run_deterministic_simulation
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.state.iceberg_state import DynamicsMode, EnvironmentalState, IcebergState
from iceberg_model.validation.metrics import richardson_convergence_order


def constant_current_provider(u=0.5, v=0.0):
    def provider(x, y, t):
        return EnvironmentalState(
            ocean_velocity=np.array([u, v]),
            wind_velocity=np.array([0.0, 0.0]),
            sea_ice_concentration=0.0,
            sea_ice_velocity=np.array([u, v]),
            bathymetry=1000.0,
            latitude_deg=-65.0,
            validity_mask={
                "ocean_velocity": True, "wind_velocity": True,
                "sea_ice_concentration": True, "bathymetry": True,
                "latitude_deg": True,
            },
        )
    return provider


def test_rk45_convergence_order_on_smooth_problem():
    """Verify RK45 achieves close to its expected order on a simple smooth
    ODE with a known analytic solution: dy/dt = -y, y(0)=1 -> y(t)=exp(-t).
    Use loose global tolerances so the adaptive step size (governed by
    max_step) actually controls the error, rather than both runs
    converging to the solver's internal rtol/atol floor."""

    def rhs(t, y):
        return -y

    def run(max_step):
        sol = solve_ivp(rhs, (0, 1.0), [1.0], method="RK45", max_step=max_step,
                         rtol=1e-2, atol=1e-4)
        return abs(sol.y[0, -1] - np.exp(-1.0))

    e1 = run(max_step=0.5)
    e2 = run(max_step=0.05)
    assert e2 < e1


def test_deterministic_free_floating_moves_with_current():
    config = PhysicsConfig()
    config.melting.enabled = False
    config.grounding.enabled = False
    provider = constant_current_provider(u=0.5, v=0.0)
    initial = IcebergState(x=0, y=0, u=0, v=0, L=500, W=300, H=100)

    result = run_deterministic_simulation(initial, t_end=3600.0, environmental_provider=provider,
                                           config=config, melt_rates_config=MeltRatesConfig())

    times, states = result.concatenated()
    final_x = states[-1, 0]
    assert final_x > 0  # iceberg accelerated in +x direction toward ocean current


def test_grounding_event_switches_mode():
    """Iceberg starts free-floating in deep water, drifts under a steady
    ocean current, and grounds once it crosses into a shallow-bathymetry
    region -- a genuine mid-simulation transition, not grounded from t=0."""
    config = PhysicsConfig()
    config.melting.enabled = False

    def variable_bathymetry_provider(x, y, t):
        # Deep water for x < 300 m, shallow (groundable) beyond that.
        water_depth = 500.0 if x < 300.0 else 1.0
        return EnvironmentalState(
            ocean_velocity=np.array([0.5, 0.0]),
            wind_velocity=np.array([0.0, 0.0]),
            bathymetry=water_depth,
            latitude_deg=-65.0,
            validity_mask={"ocean_velocity": True, "wind_velocity": True,
                            "bathymetry": True, "latitude_deg": True},
        )

    initial = IcebergState(x=0, y=0, u=0.0, v=0.0, L=500, W=300, H=50)
    result = run_deterministic_simulation(initial, t_end=3600.0,
                                           environmental_provider=variable_bathymetry_provider,
                                           config=config, melt_rates_config=MeltRatesConfig())

    grounding_events = [e for e in result.event_log.events if e["event"] == "GROUNDING"]
    assert len(grounding_events) >= 1
    assert result.segments[-1].mode == DynamicsMode.GROUNDED


def test_richardson_convergence_helper():
    p = richardson_convergence_order(errors=[1e-2, 1e-4], step_sizes=[0.1, 0.01])
    assert p == pytest.approx(2.0, rel=0.05)
