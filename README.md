# Antarctic Iceberg Drift Physics Engine

**Team Kryptonite** — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and
Navigation Decision Support System.

This repository implements the **physics-first stage** of the full
pipeline described in the master engineering prompt:

```
GeoTIFF/environmental data -> physical iceberg state -> forces ->
equations of motion -> numerical integration ->
grounding/melting/event handling -> trajectory output ->
uncertainty-ready architecture
```

The ConvLSTM / ML residual-correction stage is explicitly **out of scope**
here by design — this engine runs completely standalone
(`ResidualModel = None` is a first-class, tested configuration) and must
continue to do so even after a residual model is eventually added.

## Status

All 81 unit/integration tests pass (`pytest -q`). Three runnable examples
plus an interactive notebook demonstrate the engine end-to-end using
synthetic data (no external files required).

## Quick start

```bash
cd project
pip install -e .
pytest -q                              # run the test suite (81 tests)
python examples/synthetic_iceberg.py   # inspect mass/buoyancy for sample icebergs
python examples/synthetic_environment.py  # inspect a sampled environmental state
python examples/run_real_data.py       # full simulation + CSV export
jupyter notebook notebooks/demo.ipynb  # interactive walkthrough with plots
```

Alternative install paths: `requirements.txt` (+`requirements-optional.txt`,
`requirements-live-data.txt`) for plain pip, `environment.yml` for conda,
`Dockerfile` for a containerized run (`docker build -t iceberg-model .`).
CI runs the full suite on every push (`.github/workflows/tests.yml`).

To run against real data, replace `build_synthetic_environment()` in
`examples/run_real_data.py` with an `EnvironmentalDataset` built from your
GeoTIFF/NetCDF files via `data/geotiff_reader.py` + `data/reprojection.py`,
or use the automatic live-data pipeline described next.

## Automatic live data ingestion — no manual downloads

`data/fetchers/` + `data/live_pipeline.py` pull all four environmental
inputs directly from their source APIs (ERA5 wind via Copernicus CDS,
AMSR2 sea-ice concentration + BedMachine bathymetry via NASA
Earthdata/NSIDC, ocean currents via Copernicus Marine Service) and
assemble a ready-to-use `EnvironmentalDataset` — no manual file handling.

```bash
pip install -e ".[live-data]"
cp .env.example .env   # fill in free credentials for the 4 services
python examples/fetch_and_run_live.py \
    --min-lon -60 --max-lon -55 --min-lat -66 --max-lat -63 --date 2026-01-15
```

**Sandbox note**: this repository was built in a development environment
whose network access is restricted to package registries, so the live
fetchers could not be exercised against the real endpoints here. Each
fetcher calls its service's real client library (`cdsapi`, `earthaccess`,
`copernicusmarine`) with correct request parameters and is covered by
tests using synthetic/mocked data (`tests/test_live_data_fetchers.py`,
`tests/test_live_pipeline.py`) — run it from a machine with open network
access to validate against live data. See `docs/data_pipeline.md` for
full setup instructions and design notes.

## Corridor scenario: India's Bharati <-> Maitri resupply route

`examples/fetch_and_run_corridor.py` seeds icebergs near Bharati
(Prydz Bay), Maitri (Lazarev Sea), and the Amery Ice Shelf/Lambert
Glacier area roughly midway — the coastal stretch of NCPOR's annual
Cape Town-Bharati-Maitri-Cape Town resupply voyage that's actually
relevant to iceberg hazard (the open-ocean Cape Town leg is out of
scope — EPSG:3031 isn't meaningful that far from the pole, and icebergs
don't range that far north here). It runs all three simulations and
exports one combined GeoJSON (route + station markers + all
trajectories) for loading into QGIS or geojson.io.

```bash
python examples/fetch_and_run_corridor.py --synthetic   # test the pipeline now, no network/credentials
python examples/fetch_and_run_corridor.py --date 2026-01-15   # real ERA5/AMSR2/Copernicus/BedMachine data
```

The plotted route is an illustrative straight-line interpolation between
station coordinates, not the vessel's actual recorded track — see the
script's docstring.

## Project layout

```
src/iceberg_model/
    config/       - PhysicsConfig: every physical parameter, no hardcoding
    state/        - IcebergState, EnvironmentalState, ForceBreakdown
    data/         - GeoTIFF I/O, reprojection, spatial+temporal interpolation
    physics/      - buoyancy, drag (ocean/wind/sea-ice), Coriolis,
                    pressure gradient, grounding, melting, force aggregation,
                    equations of motion (dynamics.py)
    numerical/    - deterministic solver (RK45/DOP853/Radau + hybrid events),
                    stochastic solver (Euler-Maruyama), event definitions
    uncertainty/  - diffusion matrix, ResidualModel interface (unimplemented
                    by design), trajectory ensembles
    validation/   - physical sanity checks, convergence & trajectory metrics
    simulation/   - Simulator: the top-level orchestration class
    io/           - CSV / GeoJSON trajectory export with event metadata
tests/            - 81 tests, one file per module (see docs/validation.md)
examples/         - four runnable scripts (three synthetic, one live-data), all documented
notebooks/        - demo.ipynb: interactive walkthrough with plots
configs/          - default.yaml (+ config/loader.py to load/save it)
docs/             - physics.md, data_pipeline.md, numerical_methods.md, validation.md
```

## Packaging & deployment

- `pyproject.toml` (+ `requirements*.txt`, `environment.yml`) — pip or conda install
- `Dockerfile` / `.dockerignore` — containerized, reproducible runs
- `.github/workflows/tests.yml` — CI on every push/PR (Python 3.11 + 3.12)
- `config/loader.py` — load/save `PhysicsConfig` from/to YAML, with full validation
- `logging_config.py` — opt-in structured logging for scripts/services; live fetchers log fetch/cache activity
- `LICENSE` (MIT, with a research-tool disclaimer) / `CONTRIBUTING.md`

## Design principles followed

- **Physics-first**: nothing in `physics/dynamics.py` calls into ML.
- **Unit-safe / coordinate-system-safe**: SI units throughout; all
  dynamics happen in a projected CRS (default `EPSG:3031`), lat/lon only
  at I/O boundaries.
- **Every physical parameter is configurable** — see `config/physics_config.py`;
  nothing is hardcoded in a force module.
- **Missing data never silently becomes a valid physical value** — see
  `EnvironmentalState.validity_mask` and its use throughout `physics/*`.
- **Deterministic baseline before stochastic extension** — the SDE solver
  is a separate, later-added module; the deterministic engine is fully
  functional and tested on its own.
- **No arbitrary neural correction can violate physical bounds** — there
  is no neural correction in this codebase; the interface for one
  (`uncertainty/residuals.py::ResidualModel`) is abstract and unused.
- **No claim of certified real-world safety** — see the disclaimer
  embedded in every `io/trajectory_export.py` GeoJSON output.

See `docs/physics.md` for the full physical model writeup, including
every equation, its source/derivation, and its documented limitations.
