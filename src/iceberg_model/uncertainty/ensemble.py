"""
Ensemble trajectory generation: run N stochastic realizations to build the
trajectory ensemble that feeds downstream collision-probability / risk-map
stages (outside this task's scope, but the interface is prepared here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.dynamics import EnvironmentalProvider
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.numerical.stochastic_solver import StochasticTrajectory, run_euler_maruyama
from iceberg_model.state.iceberg_state import IcebergState
from iceberg_model.uncertainty.diffusion import DiffusionConfig


@dataclass
class TrajectoryEnsemble:
    members: list  # list[StochasticTrajectory]

    def position_spread(self, step_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_xy, std_xy) across ensemble members at a given step index."""
        pts = np.array([m.states[step_index, 0:2] for m in self.members])
        return pts.mean(axis=0), pts.std(axis=0)


def run_ensemble(
    initial_state: IcebergState,
    t_end: float,
    dt: float,
    environmental_provider: EnvironmentalProvider,
    config: PhysicsConfig,
    diffusion_config: DiffusionConfig,
    n_members: int = 50,
    melt_rates_config: Optional[MeltRatesConfig] = None,
    seed: Optional[int] = None,
) -> TrajectoryEnsemble:
    rng = np.random.default_rng(seed)
    members = []
    for _ in range(n_members):
        traj = run_euler_maruyama(
            initial_state, t_end, dt, environmental_provider, config,
            diffusion_config, melt_rates_config=melt_rates_config, rng=rng,
        )
        members.append(traj)
    return TrajectoryEnsemble(members=members)
