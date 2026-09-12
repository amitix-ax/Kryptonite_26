"""
Example: constructing synthetic IcebergState objects for quick testing
without any real data. Run with:

    python examples/synthetic_iceberg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.physics.buoyancy import calculate_freeboard, calculate_submerged_depth
from iceberg_model.physics.forces import iceberg_mass
from iceberg_model.state.iceberg_state import IcebergState


def make_tabular_iceberg(x=0.0, y=0.0, L=1200.0, W=600.0, H=180.0) -> IcebergState:
    """A large tabular Antarctic iceberg, roughly A-68-scale proportions
    (dimensions illustrative, not a real observation)."""
    return IcebergState(x=x, y=y, u=0.0, v=0.0, L=L, W=W, H=H)


def make_small_bergy_bit(x=0.0, y=0.0) -> IcebergState:
    return IcebergState(x=x, y=y, u=0.0, v=0.0, L=15.0, W=10.0, H=4.0)


if __name__ == "__main__":
    config = PhysicsConfig()

    for name, state in [
        ("tabular iceberg", make_tabular_iceberg()),
        ("bergy bit", make_small_bergy_bit()),
    ]:
        mass = iceberg_mass(state, config)
        D = calculate_submerged_depth(state, config)
        fb = calculate_freeboard(state, config)
        print(f"{name}: L={state.L}m W={state.W}m H={state.H}m "
              f"mass={mass:.3e} kg keel_depth={D:.1f}m freeboard={fb:.1f}m")
