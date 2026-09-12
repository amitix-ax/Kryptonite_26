import json

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.io.trajectory_export import export_multi_trajectory_geojson
from iceberg_model.numerical.deterministic_solver import run_deterministic_simulation
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState


def _flat_provider(x, y, t):
    return EnvironmentalState(
        ocean_velocity=np.array([0.2, 0.0]),
        wind_velocity=np.array([0.0, 0.0]),
        bathymetry=1000.0,
        latitude_deg=-69.0,
        validity_mask={"ocean_velocity": True, "wind_velocity": True,
                        "bathymetry": True, "latitude_deg": True},
    )


def test_multi_trajectory_geojson_combines_route_stations_and_icebergs(tmp_path):
    config = PhysicsConfig()
    config.melting.enabled = False

    results = {}
    for name, (x0, y0) in [("near_bharati", (2195417.0, 539758.0)), ("near_maitri", (428787.0, 2064746.0))]:
        initial = IcebergState(x=x0, y=y0, u=0.0, v=0.0, L=500, W=300, H=100)
        results[name] = run_deterministic_simulation(
            initial, t_end=3600.0, environmental_provider=_flat_provider,
            config=config, melt_rates_config=MeltRatesConfig(),
        )

    route = [(11.73, -70.77), (40.0, -69.5), (76.19, -69.41)]
    stations = {"Maitri": (11.73, -70.77), "Bharati": (76.19, -69.41)}

    out = tmp_path / "corridor.geojson"
    export_multi_trajectory_geojson(results, config, out, route_waypoints_lonlat=route, stations=stations)

    data = json.loads(out.read_text())
    kinds = [f["properties"]["kind"] for f in data["features"]]
    assert kinds.count("route") == 1
    assert kinds.count("station") == 2
    assert kinds.count("iceberg_trajectory") >= 2

    iceberg_names = {f["properties"]["iceberg"] for f in data["features"] if f["properties"]["kind"] == "iceberg_trajectory"}
    assert iceberg_names == {"near_bharati", "near_maitri"}
    assert "near_bharati" in data["metadata"]["icebergs"]
    assert "near_maitri" in data["metadata"]["icebergs"]


def test_multi_trajectory_geojson_works_without_route_or_stations(tmp_path):
    config = PhysicsConfig()
    config.melting.enabled = False
    initial = IcebergState(x=0, y=0, u=0.0, v=0.0, L=500, W=300, H=100)
    result = run_deterministic_simulation(initial, t_end=3600.0, environmental_provider=_flat_provider,
                                           config=config, melt_rates_config=MeltRatesConfig())

    out = tmp_path / "single.geojson"
    export_multi_trajectory_geojson({"only_berg": result}, config, out)

    data = json.loads(out.read_text())
    assert all(f["properties"]["kind"] == "iceberg_trajectory" for f in data["features"])
