"""
Example: fetch REAL environmental data directly from ERA5, AMSR2/NSIDC,
Copernicus Marine, and BedMachine Antarctica, then run the physics engine
against it -- no manually downloaded files.

Requires:
    pip install -e ".[live-data]"
    Credentials set per .env.example / docs/data_pipeline.md.
    Outbound network access to cds.climate.copernicus.eu,
    urs.earthdata.nasa.gov, and data.marine.copernicus.eu.

This CANNOT be run from the development sandbox that built this
repository (its network egress is restricted to package registries) --
run it from your own machine/server/cloud VM once credentials are set up.

    python examples/fetch_and_run_live.py --min-lon -60 --max-lon -55 \\
        --min-lat -66 --max-lat -63 --date 2026-01-15
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.data.fetchers.common import BoundingBox, DiskCache, TimeRange
from iceberg_model.data.fetchers.credentials import load_dotenv_if_present
from iceberg_model.data.live_pipeline import LiveEnvironmentBuilder
from iceberg_model.data.reprojection import build_common_grid
from iceberg_model.io.trajectory_export import export_csv, export_geojson
from iceberg_model.physics.melting import MeltRatesConfig
from iceberg_model.simulation.simulator import Simulator
from iceberg_model.state.iceberg_state import IcebergState
from iceberg_model.validation.physical_checks import check_state

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD, start of a 1-day fetch window")
    parser.add_argument("--resolution-m", type=float, default=5000.0)
    parser.add_argument("--sim-hours", type=float, default=24.0)
    parser.add_argument("--start-x", type=float, default=None, help="iceberg start x [m] in EPSG:3031 (default: center of bbox)")
    parser.add_argument("--start-y", type=float, default=None, help="iceberg start y [m] in EPSG:3031 (default: center of bbox)")
    parser.add_argument("--length-m", type=float, default=800.0)
    parser.add_argument("--width-m", type=float, default=400.0)
    parser.add_argument("--thickness-m", type=float, default=150.0)
    args = parser.parse_args()

    # Enable logging so fetcher cascade (AMSR2 -> ERA5 -> SIC=0) messages
    # are visible — without this, logger.info/warning from fetchers is
    # silently swallowed.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    load_dotenv_if_present()

    config = PhysicsConfig()
    bbox = BoundingBox(min_lon=args.min_lon, min_lat=args.min_lat,
                        max_lon=args.max_lon, max_lat=args.max_lat)
    start = datetime.fromisoformat(args.date)
    time_range = TimeRange(start=start, end=start + timedelta(days=1))

    # Auto-compute iceberg start position from bbox center if not
    # explicitly provided (avoids the (0,0) = South Pole default which
    # is outside any realistic data grid).
    from pyproj import Transformer
    to_working = Transformer.from_crs(
        config.crs.geographic_crs, config.crs.working_crs, always_xy=True
    )
    center_lon = (args.min_lon + args.max_lon) / 2
    center_lat = (args.min_lat + args.max_lat) / 2
    cx, cy = to_working.transform(center_lon, center_lat)
    start_x = args.start_x if args.start_x is not None else cx
    start_y = args.start_y if args.start_y is not None else cy
    print(f"Iceberg start position: ({start_x:.0f}, {start_y:.0f}) m in {config.crs.working_crs}")
    print(f"  (lon={center_lon:.2f}, lat={center_lat:.2f} -> projected)")

    # Build a working-CRS grid covering roughly the requested bbox. A
    # production caller would derive tighter bounds from the bbox's
    # actual EPSG:3031 projection rather than this fixed-size default.
    grid = build_common_grid(
        crs=config.crs.working_crs,
        resolution_m=args.resolution_m,
        bounds=(-2_800_000, -2_800_000, 2_800_000, 2_800_000),
    )

    print("Fetching live environmental data (ERA5, AMSR2/NSIDC, Copernicus Marine, BedMachine)...")
    builder = LiveEnvironmentBuilder(grid=grid, cache=DiskCache())
    environment = builder.build(bbox, time_range)
    print("Live environment built.")

    # Report which SIC source is active.
    sic_field = environment.temporal_fields.get("sea_ice_concentration")
    if sic_field is not None:
        import numpy as np
        all_vals = np.concatenate([f.ravel() for f in sic_field.fields])
        finite = all_vals[np.isfinite(all_vals)]
        print(f"Sea-ice concentration: {len(sic_field.fields)} timestep(s), "
              f"mean={np.mean(finite):.4f}, max={np.max(finite):.4f}")
    else:
        print("WARNING: No sea-ice concentration field loaded — SIC drag will be ZERO.")

    # Live data temporal fields use epoch seconds (e.g. 1768435200.0 for
    # 2026-01-15). The solver queries environment at t=state.time, so we
    # must set the initial time and t_end in epoch seconds too — otherwise
    # t=0.0 means 1970-01-01 and every temporal sample returns None.
    t_start_epoch = start.timestamp()

    simulator = Simulator(config=config, environment=environment, melt_rates_config=MeltRatesConfig())
    initial_state = IcebergState(
        x=start_x, y=start_y, u=0.0, v=0.0,
        L=args.length_m, W=args.width_m, H=args.thickness_m,
        time=t_start_epoch,
    )
    result = simulator.run(initial_state, t_end=t_start_epoch + args.sim_hours * 3600.0)

    times, states = result.concatenated()
    final_state = IcebergState.from_vector(states[-1], time=times[-1], mode=result.final_mode)
    report = check_state(final_state, config, water_depth_m=None)

    print(f"Simulated {len(times)} steps over {args.sim_hours} hours.")
    print(f"Final mode: {result.final_mode.value}")
    print(f"Events: {result.event_log.events}")
    print(f"Physical check: ok={report.ok}, violations={report.violations}")

    out_dir = Path(__file__).resolve().parents[1] / "data"
    export_csv(result, out_dir / "live_trajectory.csv")
    export_geojson(result, out_dir / "live_trajectory.geojson", config)

    from iceberg_model.io.trajectory_export import export_dashboard_json
    export_dashboard_json(
        result, config, out_dir / "dashboard_data.json",
        data_sources={"wind": "ERA5", "sea_ice": "ERA5 fallback", "currents": "Copernicus Marine", "bathymetry": "BedMachine"},
    )
    print(f"Trajectory written to {out_dir}/live_trajectory.{{csv,geojson}}")
    print(f"Dashboard data written to {out_dir}/dashboard_data.json")


if __name__ == "__main__":
    main()
