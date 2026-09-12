"""
Load a PhysicsConfig from a YAML file (e.g. configs/default.yaml).

Kept separate from physics_config.py itself so that module has zero I/O
dependencies (no PyYAML import) and can be constructed purely in-memory,
as every test in this repository does. This loader is the only place
YAML enters the picture.
"""

from __future__ import annotations

from pathlib import Path

from iceberg_model.config.physics_config import PhysicsConfig


def load_physics_config(path: str | Path) -> PhysicsConfig:
    """
    Parse a YAML file into a validated PhysicsConfig. Pydantic validation
    (including the rho_ice < rho_seawater sanity check) runs exactly as
    it would for a config built in Python -- a malformed or physically
    invalid YAML file raises the same errors, not a silently-ignored one.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load YAML configs. Install with `pip install pyyaml`.") from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file {path} is empty.")

    return PhysicsConfig(**raw)


def save_physics_config(config: PhysicsConfig, path: str | Path) -> None:
    """Write a PhysicsConfig back out to YAML (e.g. to snapshot the exact
    configuration used for a simulation run, alongside its trajectory output)."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to save YAML configs. Install with `pip install pyyaml`.") from e

    path = Path(path)
    with open(path, "w") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)
