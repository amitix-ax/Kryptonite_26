"""
`sample_environment(x, y, t)` (spec section 9): the single entry point that
turns raw gridded fields (already reprojected onto a common grid) into a
typed EnvironmentalState at a specific iceberg position and time.

A TIFF/NetCDF/xarray field is a spatial field E(x, y, t), not a trajectory.
This module is the bridge: spatial interpolation (interpolation.py) +
temporal interpolation (temporal_loader.py) -> EnvironmentalState with an
explicit validity_mask, per principle J.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.data.interpolation import bilinear_interpolate
from iceberg_model.data.temporal_loader import TemporalField, interpolate_in_time, TimeOutOfBoundsError
from iceberg_model.state.iceberg_state import EnvironmentalState


@dataclass
class EnvironmentalDataset:
    """
    Holds one TemporalField (or a single static field) per named variable,
    all pre-reprojected onto the same CommonGrid (see data/reprojection.py),
    plus that grid's affine transform for spatial interpolation.
    """

    transform: tuple  # CommonGrid.transform
    temporal_fields: dict = field(default_factory=dict)   # name -> TemporalField
    static_fields: dict = field(default_factory=dict)     # name -> np.ndarray (e.g. bathymetry)
    sea_ice_velocity_field: Optional[str] = None           # name of a temporal_field to use, or None
    latitude_field: Optional[np.ndarray] = None             # static lat grid for Coriolis, or None

    def _sample_temporal(self, name: str, x: float, y: float, t: float,
                          allow_extrapolation: bool) -> tuple[Optional[float], bool, str]:
        tf: Optional[TemporalField] = self.temporal_fields.get(name)
        if tf is None:
            return None, False, "field_not_loaded"
        try:
            field_at_t = interpolate_in_time(tf, t, allow_extrapolation=allow_extrapolation)
        except TimeOutOfBoundsError:
            return None, False, "time_out_of_bounds"

        result = bilinear_interpolate(field_at_t, self.transform, x, y)
        if not result.valid:
            return None, False, result.reason
        return result.value, True, ""

    def _sample_static(self, name: str, x: float, y: float) -> tuple[Optional[float], bool, str]:
        arr = self.static_fields.get(name)
        if arr is None:
            return None, False, "field_not_loaded"
        result = bilinear_interpolate(arr, self.transform, x, y)
        if not result.valid:
            return None, False, result.reason
        return result.value, True, ""

    def _sample_ssh_gradient(self, x: float, y: float, t: float,
                              allow_extrapolation: bool) -> tuple[Optional[np.ndarray], bool]:
        """
        Central finite-difference estimate of grad_h(SSH) at (x, y, t),
        using the same TemporalField -> temporal interpolation ->
        spatial interpolation path as every other field (no separate
        code path or fabricated derivative). Step size defaults to one
        grid cell (`transform`'s x-resolution) so the finite difference
        stays local to the data's actual resolution.

        Returns (None, False) -- never a fabricated gradient -- if any of
        the four surrounding SSH samples is unavailable (out of bounds,
        nodata, or the field isn't loaded at all).
        """
        xres = abs(self.transform[0])
        yres = abs(self.transform[4])

        e, e_ok, _ = self._sample_temporal("sea_surface_height", x + xres, y, t, allow_extrapolation)
        w, w_ok, _ = self._sample_temporal("sea_surface_height", x - xres, y, t, allow_extrapolation)
        n, n_ok, _ = self._sample_temporal("sea_surface_height", x, y + yres, t, allow_extrapolation)
        s, s_ok, _ = self._sample_temporal("sea_surface_height", x, y - yres, t, allow_extrapolation)

        if not (e_ok and w_ok and n_ok and s_ok):
            return None, False

        d_dx = (e - w) / (2 * xres)
        d_dy = (n - s) / (2 * yres)
        return np.array([d_dx, d_dy]), True

    def sample_environment(self, x: float, y: float, t: float, config: PhysicsConfig) -> EnvironmentalState:
        allow_extrap = config.temporal_interpolation.allow_extrapolation
        validity: dict[str, bool] = {}

        ou, ou_ok, _ = self._sample_temporal("ocean_u", x, y, t, allow_extrap)
        ov, ov_ok, _ = self._sample_temporal("ocean_v", x, y, t, allow_extrap)
        ocean_velocity = np.array([ou, ov]) if (ou_ok and ov_ok) else None
        validity["ocean_velocity"] = ou_ok and ov_ok

        wu, wu_ok, _ = self._sample_temporal("wind_u", x, y, t, allow_extrap)
        wv, wv_ok, _ = self._sample_temporal("wind_v", x, y, t, allow_extrap)
        wind_velocity = np.array([wu, wv]) if (wu_ok and wv_ok) else None
        validity["wind_velocity"] = wu_ok and wv_ok

        sic, sic_ok, _ = self._sample_temporal("sea_ice_concentration", x, y, t, allow_extrap)
        validity["sea_ice_concentration"] = sic_ok

        sst, sst_ok, _ = self._sample_temporal("sea_surface_temperature", x, y, t, allow_extrap)
        validity["sst"] = sst_ok

        ssh, ssh_ok, _ = self._sample_temporal("sea_surface_height", x, y, t, allow_extrap)
        validity["ssh"] = ssh_ok

        ssh_gradient = None
        if config.pressure_gradient.enabled and "sea_surface_height" in self.temporal_fields:
            ssh_gradient, ssh_grad_ok = self._sample_ssh_gradient(x, y, t, allow_extrap)
            validity["ssh_gradient"] = ssh_grad_ok
        else:
            validity["ssh_gradient"] = False

        bathy, bathy_ok, _ = self._sample_static("bathymetry", x, y)
        validity["bathymetry"] = bathy_ok

        lat = None
        if self.latitude_field is not None:
            lat_result = bilinear_interpolate(self.latitude_field, self.transform, x, y)
            lat = lat_result.value if lat_result.valid else None
        validity["latitude_deg"] = lat is not None

        # Sea-ice velocity: explicit field if configured, else fallback per
        # config.sea_ice_drag.fallback_velocity_source (never fabricated,
        # always flagged).
        sea_ice_velocity = None
        is_fallback = False
        if self.sea_ice_velocity_field:
            siu, siu_ok, _ = self._sample_temporal(f"{self.sea_ice_velocity_field}_u", x, y, t, allow_extrap)
            siv, siv_ok, _ = self._sample_temporal(f"{self.sea_ice_velocity_field}_v", x, y, t, allow_extrap)
            if siu_ok and siv_ok:
                sea_ice_velocity = np.array([siu, siv])
        if sea_ice_velocity is None and ocean_velocity is not None and \
                config.sea_ice_drag.fallback_velocity_source == "ocean_current":
            sea_ice_velocity = ocean_velocity.copy()
            is_fallback = True

        return EnvironmentalState(
            ocean_velocity=ocean_velocity,
            wind_velocity=wind_velocity,
            sea_ice_concentration=sic,
            sea_ice_velocity=sea_ice_velocity,
            sst=sst,
            ssh=ssh,
            ssh_gradient=ssh_gradient,
            bathymetry=bathy,
            latitude_deg=lat,
            validity_mask=validity,
            sea_ice_velocity_is_fallback=is_fallback,
        )
