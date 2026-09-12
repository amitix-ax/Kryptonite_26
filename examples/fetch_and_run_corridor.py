"""
Simulate iceberg drift along the Bharati <-> Maitri coastal corridor --
the Antarctic-side stretch of India's Cape Town - Bharati - Maitri -
Cape Town resupply voyage (NCPOR's annual expedition aboard the chartered
ice-class vessel MV Vasiliy Golovnin or equivalent).

Scope note: this script covers the ANTARCTIC COASTAL portion of the
route only -- roughly the 3,000 km stretch between the two stations. It
deliberately does NOT extend to Cape Town: EPSG:3031 (Antarctic Polar
Stereographic, this engine's working CRS) is only valid/meaningful near
the pole, and icebergs of the kind this engine models don't range up to
Cape Town's latitude anyway. The coastal stretch between Bharati and
Maitri is also the scientifically relevant part for iceberg hazard --
it includes Prydz Bay / the Amery Ice Shelf area near Bharati, a known
iceberg-calving region the voyage transits.

The plotted "route" is an ILLUSTRATIVE interpolation between the two
station coordinates along the coast (piecewise-linear in lon/lat) -- it
is NOT the vessel's actual recorded GPS track. No public API for the
real track is queried here. Treat it as a visual reference for "roughly
where the ship goes," not as navigational data.

Two modes:
    --synthetic   Uses made-up but internally-consistent environmental
                  fields (like examples/synthetic_environment.py) so this
                  script runs immediately, with no credentials and no
                  network access. Use this to verify the corridor/
                  multi-iceberg/export logic before setting up live data.
    (default)     Fetches REAL ERA5 wind, AMSR2 sea ice, Copernicus
                  Marine currents, and BedMachine bathymetry for the
                  whole corridor via LiveEnvironmentBuilder. Requires
                  `pip install -e ".[live-data]"`, the credentials in
                  .env (see .env.example / docs/data_pipeline.md), and
                  outbound network access this sandbox does not have.

Usage:
    python examples/fetch_and_run_corridor.py --synthetic
    python examples/fetch_and_run_corridor.py --date 2026-01-15
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.data.environmental_dataset import EnvironmentalDataset
from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.credentials import load_dotenv_if_present
from iceberg_model.data.live_pipeline import LiveEnvironmentBuilder
from iceberg_model.data.reprojection import CommonGrid, build_common_grid
from iceberg_model.data.temporal_loader import TemporalField
from iceberg_model.io.trajectory_export import export_multi_trajectory_geojson
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.simulation.simulator import Simulator
from iceberg_model.state.iceberg_state import IcebergState
from iceberg_model.validation.physical_checks import check_state

# India's two active Antarctic research stations (lon, lat, degrees).
STATIONS = {
    "Maitri": (11.7319, -70.7667),   # Schirmacher Oasis, Queen Maud Land
    "Bharati": (76.1874, -69.4080),  # Larsemann Hills, Prydz Bay
}

# Iceberg seed points along the corridor: near each station, plus one
# roughly midway (Amery Ice Shelf / Lambert Glacier sector, a major
# calving source the voyage transits between the two stations).
ICEBERG_SEEDS = {
    "near_bharati": (78.0, -68.5),
    "midway_amery": (68.0, -67.0),
    "near_maitri": (9.0, -69.5),
}


def build_corridor_bbox(margin_deg: float = 3.0) -> BoundingBox:
    lons = [lon for lon, lat in STATIONS.values()] + [lon for lon, lat in ICEBERG_SEEDS.values()]
    lats = [lat for lon, lat in STATIONS.values()] + [lat for lon, lat in ICEBERG_SEEDS.values()]
    return BoundingBox(
        min_lon=min(lons) - margin_deg, max_lon=max(lons) + margin_deg,
        min_lat=min(lats) - margin_deg, max_lat=max(lats) + margin_deg,
    )


def build_corridor_grid(config: PhysicsConfig, resolution_m: float = 15_000.0) -> CommonGrid:
    """
    A working-CRS grid sized to comfortably contain both stations plus
    margin. Sized empirically (not ±2,000,000 m, which is too small --
    Bharati alone projects to ~2.2 million m from the pole in EPSG:3031).
    """
    return build_common_grid(
        crs=config.crs.working_crs,
        resolution_m=resolution_m,
        bounds=(-2_800_000, -2_800_000, 2_800_000, 2_800_000),
    )


def build_synthetic_corridor_environment(grid: CommonGrid) -> EnvironmentalDataset:
    """
    Offline stand-in for LiveEnvironmentBuilder, covering the same grid,
    for testing the corridor/multi-iceberg/export pipeline without
    network access or credentials. Fields are simple, spatially-varying
    but NOT observationally grounded -- do not use this for anything
    beyond validating the pipeline's plumbing.
    """
    shape = (grid.height, grid.width)
    xres, _, x0, _, yres, y0 = grid.transform
    xs = x0 + xres * (np.arange(grid.width) + 0.5)
    ys = y0 + yres * (np.arange(grid.height) + 0.5)
    xx, yy = np.meshgrid(xs, ys)

    # A mild coast-following current: eastward, strength decaying with
    # distance from the pole (a crude proxy for the Antarctic Coastal
    # Current, purely illustrative).
    r = np.sqrt(xx**2 + yy**2)
    r_norm = np.clip(r / 2_500_000.0, 0.2, 1.0)
    ocean_u = 0.4 * (1.0 - 0.3 * r_norm)
    ocean_v = np.zeros(shape)
    wind_u = np.full(shape, 0.0)
    wind_v = np.full(shape, -3.0)
    sic = np.clip(1.0 - r_norm, 0.0, 1.0)
    sst = np.full(shape, 0.5)
    bathymetry = np.clip(800.0 - 0.0004 * (r - 2_000_000.0), 20.0, 4000.0)

    from pyproj import Transformer
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    _, latitude = transformer.transform(xx, yy)

    return EnvironmentalDataset(
        transform=grid.transform,
        temporal_fields={
            "ocean_u": TemporalField(timestamps=[0.0, 1e7], fields=[ocean_u, ocean_u]),
            "ocean_v": TemporalField(timestamps=[0.0, 1e7], fields=[ocean_v, ocean_v]),
            "wind_u": TemporalField(timestamps=[0.0, 1e7], fields=[wind_u, wind_u]),
            "wind_v": TemporalField(timestamps=[0.0, 1e7], fields=[wind_v, wind_v]),
            "sea_ice_concentration": TemporalField(timestamps=[0.0, 1e7], fields=[sic, sic]),
            "sea_surface_temperature": TemporalField(timestamps=[0.0, 1e7], fields=[sst, sst]),
        },
        static_fields={"bathymetry": bathymetry},
        latitude_field=latitude,
    )


def build_route_waypoints(n_per_leg: int = 8) -> list[tuple[float, float]]:
    """Piecewise-linear illustrative route through Bharati -> midway seeds
    -> Maitri, sorted roughly east-to-west along the coast. See module
    docstring: this is NOT a recorded vessel track."""
    ordered_points = [
        STATIONS["Bharati"],
        ICEBERG_SEEDS["midway_amery"],
        STATIONS["Maitri"],
    ]
    waypoints = []
    for (lon0, lat0), (lon1, lat1) in zip(ordered_points[:-1], ordered_points[1:]):
        for i in range(n_per_leg):
            t = i / n_per_leg
            waypoints.append((lon0 + t * (lon1 - lon0), lat0 + t * (lat1 - lat0)))
    waypoints.append(ordered_points[-1])
    return waypoints


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true",
                         help="Use offline synthetic data instead of live fetches (no network/credentials needed).")
    parser.add_argument("--date", type=str, default="2026-01-15", help="YYYY-MM-DD, start of a 1-day live-fetch window")
    parser.add_argument("--sim-hours", type=float, default=72.0)
    parser.add_argument("--length-m", type=float, default=600.0)
    parser.add_argument("--width-m", type=float, default=350.0)
    parser.add_argument("--thickness-m", type=float, default=120.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )
    load_dotenv_if_present()
    config = PhysicsConfig()
    grid = build_corridor_grid(config)

    if args.synthetic:
        print("Using synthetic offline environment (--synthetic). No network access used.")
        environment = build_synthetic_corridor_environment(grid)
    else:
        print("Fetching LIVE environmental data for the Bharati<->Maitri corridor "
              "(ERA5, AMSR2/NSIDC, Copernicus Marine, BedMachine)...")
        bbox = build_corridor_bbox()
        start = datetime.fromisoformat(args.date)
        time_range = TimeRange(start=start, end=start + timedelta(days=1))
        environment = LiveEnvironmentBuilder(grid=grid, cache=DiskCache()).build(bbox, time_range)
        print("Live environment built.")

    from pyproj import Transformer
    to_working = Transformer.from_crs(config.crs.geographic_crs, config.crs.working_crs, always_xy=True)

    # Synthetic temporal fields start at t=0; live fields use epoch seconds.
    if args.synthetic:
        t_start = 0.0
    else:
        t_start = datetime.fromisoformat(args.date).timestamp()

    simulator = Simulator(config=config, environment=environment, melt_rates_config=MeltRatesConfig())
    results = {}

    for name, (lon, lat) in ICEBERG_SEEDS.items():
        x0, y0 = to_working.transform(lon, lat)
        initial_state = IcebergState(x=x0, y=y0, u=0.0, v=0.0,
                                      L=args.length_m, W=args.width_m, H=args.thickness_m,
                                      time=t_start)
        result = simulator.run(initial_state, t_end=t_start + args.sim_hours * 3600.0)
        results[name] = result

        times, states = result.concatenated()
        final_state = IcebergState.from_vector(states[-1], time=times[-1], mode=result.final_mode)
        report = check_state(final_state, config, water_depth_m=None)
        displacement_km = np.linalg.norm(states[-1, 0:2] - states[0, 0:2]) / 1000
        print(f"[{name}] seeded at (lon={lon}, lat={lat}): "
              f"{len(times)} steps, final_mode={result.final_mode.value}, "
              f"displacement={displacement_km:.1f} km, "
              f"physical_check_ok={report.ok} (violations={report.violations})")
        if result.event_log.events:
            print(f"  events: {result.event_log.events}")

    route = build_route_waypoints()
    out_path = Path(__file__).resolve().parents[1] / "data" / "corridor_trajectories.geojson"
    export_multi_trajectory_geojson(
        results, config, out_path,
        route_waypoints_lonlat=route,
        route_label="Bharati - Amery/Lambert - Maitri (illustrative)",
        stations=STATIONS,
    )
    print(f"\nCombined route + station markers + {len(results)} iceberg trajectories written to:\n  {out_path}")
    print("Open it in QGIS, geojson.io, or any GIS tool that reads GeoJSON.")


if __name__ == "__main__":
    main()
