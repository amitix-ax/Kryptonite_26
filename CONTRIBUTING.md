# Contributing

## Setup

```bash
git clone <repo>
cd project
pip install -e ".[live-data,optional]"
pytest -q
```

## Before opening a PR

- `pytest -q` must pass (currently 70 tests, ~1 second).
- Any new physics term goes through `physics/forces.py::calculate_total_acceleration`
  and gets its own `ForceBreakdown` field -- don't fold a new force into
  an existing one.
- Any new configurable value goes into `config/physics_config.py` with a
  docstring stating whether it's a sourced literature value or an
  "UNVALIDATED DEFAULT" that needs calibration. Never hardcode a
  coefficient directly in a `physics/*.py` module (see `docs/physics.md`
  and engineering principle G in the original spec).
- Any code path that could receive missing/out-of-bounds data must
  return `None` / set `validity_mask[...] = False` rather than
  substituting a plausible-looking value (principle J). If you're
  tempted to write `value or 0.0` for an environmental field, don't --
  that's exactly the silent-substitution pattern this codebase avoids.
- New tests go in the file matching the module they exercise (`test_buoyancy.py`
  for `physics/buoyancy.py`, etc.) — see `docs/validation.md` for the
  current file-to-module mapping.
- Update `docs/physics.md` / `docs/data_pipeline.md` / `docs/numerical_methods.md`
  if you change what an equation means or how data flows, not just the code.

## Style

- Type hints on public function signatures.
- No bare `except:` -- catch specific exceptions.
- Prefer raising a clear, actionable exception over returning `None`
  silently for anything that isn't an *expected* missing-data case
  (missing data uses the `validity_mask` pattern above instead).
- `ruff check src/ tests/ examples/` is run in CI as informational
  (non-blocking) for now; fixing warnings it raises is welcome but not
  required to merge.

## Scope boundary

This repository implements the physics-first pipeline stages only:
GeoTIFF/NetCDF ingestion -> physical iceberg state -> forces -> equations
of motion -> numerical integration -> grounding/melting/event handling ->
trajectory output -> uncertainty-ready architecture. ConvLSTM residual
correction, collision probability, risk maps, and A* route planning are
explicitly out of scope for this codebase (see README.md) -- PRs adding
those should go in a separate downstream repository/module that imports
this one, not into `src/iceberg_model/`.
