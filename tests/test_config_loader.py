import pytest

yaml = pytest.importorskip("yaml")

from iceberg_model.config.loader import load_physics_config, save_physics_config
from iceberg_model.config.physics_config import PhysicsConfig


def test_load_default_yaml():
    config = load_physics_config("configs/default.yaml")
    assert config.crs.working_crs == "EPSG:3031"
    assert config.solver.kind.value == "RK45"


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_physics_config("configs/does_not_exist.yaml")


def test_load_empty_yaml_raises(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ValueError):
        load_physics_config(p)


def test_load_invalid_physics_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("fluids:\n  rho_ice: 1100.0\n  rho_seawater: 1027.0\n")
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        load_physics_config(p)


def test_save_and_reload_round_trip(tmp_path):
    config = PhysicsConfig()
    out = tmp_path / "roundtrip.yaml"
    save_physics_config(config, out)
    reloaded = load_physics_config(out)
    assert reloaded == config
