"""
High-level orchestration: the one class a user/notebook/example script is
expected to actually instantiate. Wires together PhysicsConfig, an
EnvironmentalDataset, and the deterministic solver into a single
`run()` call. The ResidualModel hook is accepted but optional and unused
by default (spec section 25: physics engine must work with
ResidualModel = None).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.data.environmental_dataset import EnvironmentalDataset
from iceberg_model.numerical.deterministic_solver import SimulationResult, run_deterministic_simulation
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.state.iceberg_state import IcebergState
from iceberg_model.uncertainty.residuals import ResidualModel


@dataclass
class Simulator:
    config: PhysicsConfig
    environment: EnvironmentalDataset
    melt_rates_config: Optional[MeltRatesConfig] = None
    residual_model: Optional[ResidualModel] = None  # NOT invoked by run(); see class docstring
    dataset_bounds: Optional[tuple] = None

    def run(self, initial_state: IcebergState, t_end: float) -> SimulationResult:
        def environmental_provider(x: float, y: float, t: float):
            return self.environment.sample_environment(x, y, t, self.config)

        return run_deterministic_simulation(
            initial_state=initial_state,
            t_end=t_end,
            environmental_provider=environmental_provider,
            config=self.config,
            melt_rates_config=self.melt_rates_config or MeltRatesConfig(),
            dataset_bounds=self.dataset_bounds,
        )
