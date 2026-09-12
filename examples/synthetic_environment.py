"""
Example: building a synthetic EnvironmentalDataset in memory (no GeoTIFF
files needed) -- the same construction used by tests/test_end_to_end.py,
extracted here as a standalone reference for how to wire up fields.

Run with:
    python examples/synthetic_environment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iceberg_model.data.environmental_dataset import EnvironmentalDataset
from iceberg_model.data.temporal_loader import TemporalField


def build_synthetic_environment(grid_size: int = 50, resolution_m: float = 2000.0) -> EnvironmentalDataset:
    shape = (grid_size, grid_size)
    half_extent = grid_size * resolution_m / 2
    # transform = (xres, 0, x_origin, 0, yres[negative], y_origin) -- matches
    # data/reprojection.py::CommonGrid.transform convention (north-up grid).
    transform = (resolution_m, 0.0, -half_extent, 0.0, -resolution_m, half_extent)

    ocean_u = np.full(shape, 0.3)   # steady 0.3 m/s eastward current
    ocean_v = np.zeros(shape)
    wind_u = np.zeros(shape)
    wind_v = np.full(shape, -4.0)   # steady 4 m/s northerly wind (blowing south)
    sic = np.clip(1.0 - np.linspace(0, 1, grid_size)[None, :].repeat(grid_size, axis=0), 0, 1)
    sst = np.full(shape, 1.5)
    bathymetry = np.full(shape, 500.0)
    latitude = np.full(shape, -65.0)  # Southern Ocean

    return EnvironmentalDataset(
        transform=transform,
        temporal_fields={
            "ocean_u": TemporalField(timestamps=[0.0, 1e6], fields=[ocean_u, ocean_u]),
            "ocean_v": TemporalField(timestamps=[0.0, 1e6], fields=[ocean_v, ocean_v]),
            "wind_u": TemporalField(timestamps=[0.0, 1e6], fields=[wind_u, wind_u]),
            "wind_v": TemporalField(timestamps=[0.0, 1e6], fields=[wind_v, wind_v]),
            "sea_ice_concentration": TemporalField(timestamps=[0.0, 1e6], fields=[sic, sic]),
            "sea_surface_temperature": TemporalField(timestamps=[0.0, 1e6], fields=[sst, sst]),
        },
        static_fields={"bathymetry": bathymetry},
        latitude_field=latitude,
    )


if __name__ == "__main__":
    from iceberg_model.config.physics_config import PhysicsConfig

    env = build_synthetic_environment()
    config = PhysicsConfig()
    sample = env.sample_environment(x=0.0, y=0.0, t=0.0, config=config)
    print("Sampled environment at (0,0,t=0):")
    print(f"  ocean_velocity = {sample.ocean_velocity}")
    print(f"  wind_velocity  = {sample.wind_velocity}")
    print(f"  sea_ice_conc   = {sample.sea_ice_concentration}")
    print(f"  sst            = {sample.sst}")
    print(f"  bathymetry     = {sample.bathymetry}")
    print(f"  latitude_deg   = {sample.latitude_deg}")
    print(f"  validity_mask  = {sample.validity_mask}")
