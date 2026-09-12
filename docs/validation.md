# Validation

## Physical checks

`validation/physical_checks.py::check_state()` runs sanity assertions
against a simulated `IcebergState`:

- Positive geometry (`L, W, H > 0`).
- `0 < rho_ice < rho_seawater`.
- `0 < D < water_depth` while `FREE_FLOATING` (flags a state that should
  have triggered grounding but didn't).
- Drift speed below a suspicious-value threshold (5 m/s) — not a hard
  physical limit, but a flag that something (units, forcing, coefficients)
  is likely wrong.
- All state values finite (catches NaN/Inf propagation early).

This is run in `examples/run_real_data.py` after every simulation and is
recommended as a post-simulation gate before trusting any output.

## Numerical convergence

`validation/metrics.py::richardson_convergence_order()` — see
`docs/numerical_methods.md`.

## Trajectory metrics

`validation/trajectory_metrics.py` compares simulated tracks against
observed tracks (e.g. from satellite-derived iceberg positions):

- `position_error_series()` — Euclidean distance per matched timestamp.
- `cumulative_along_track_skill()` — skill score against a naive
  persistence baseline (1.0 = perfect, 0.0 = no better than "iceberg
  didn't move", negative = worse than persistence).

These are intended for offline validation against real tracked-iceberg
datasets once available; they are not exercised against real data in this
task (no such dataset was supplied).

## Test suite

`tests/` covers, per module: buoyancy (`test_buoyancy.py`), drag
(`test_drag.py`), Coriolis sign convention (`test_coriolis.py`),
grounding (`test_grounding.py`), melting (`test_melting.py`), GeoTIFF I/O
(`test_geotiff.py`), spatial/temporal interpolation
(`test_interpolation.py`), the deterministic solver including a live
grounding-event transition (`test_solver.py`), unit-consistency sanity
checks (`test_units.py`), and a full synthetic end-to-end run through
`Simulator` with CSV export (`test_end_to_end.py`). All 46 tests pass as
of this writing (`pytest -q` from the project root, with `src/` on
`PYTHONPATH` per `pyproject.toml`).

## What is explicitly NOT validated here

- No calibration against real Antarctic iceberg track observations — all
  drag/melt coefficients are literature-typical defaults (see
  `docs/physics.md`), not tuned.
- No comparison against an established reference model (e.g. an
  operational iceberg drift forecast product).
- This is a research/prototype decision-support system; it carries no
  certification and makes no claim of guaranteed real-world navigational
  safety (spec principles M, N).
