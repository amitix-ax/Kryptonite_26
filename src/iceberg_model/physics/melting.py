"""
Melting model (spec section 18). Deliberately kept separate from the force
equations in forces.py/dynamics.py -- melting affects geometry (dL/dt, dW/dt,
dH/dt), not directly the momentum equations, though geometry feeds back into
mass/area on the next step.

    dH/dt = -M_basal
    dL/dt = -(M_lateral + M_wave)
    dW/dt = -(M_lateral + M_wave)

Model source / units / validity regime (documented per spec requirement):

Basal melt: a widely used empirical form is a bulk turbulent heat-transfer
parameterization driven by the ocean-iceberg thermal/velocity difference,
of the general shape used by Bigg et al. (1997) and similar Antarctic
iceberg drift studies:

    M_basal = A + B * |U_rel|^C * (T_w - T_i)

where A, B, C are empirical regression coefficients (UNVALIDATED DEFAULTS
here — must be calibrated/sourced against a specific dataset before any
quantitative use), |U_rel| is the ocean-iceberg relative speed [m/s],
T_w is sea surface temperature [C], T_i is iceberg temperature (assumed
0 C at the melting interface). Units: M_basal in [m/day], converted to
[m/s] internally.

Lateral (side) melt: temperature-driven, independent of current speed,
commonly parameterized as linear or weak-power-law in (T_w - T_i):

    M_lateral = k_lateral * (T_w - T_i)      [m/day] -> [m/s]

Wave erosion: wind-wave-driven notch/calving erosion, parameterized here
as proportional to wind speed (a common simplified proxy for sea state)
above a threshold:

    M_wave = k_wave * max(0, |U_wind| - wave_threshold_speed)   [m/day] -> [m/s]

ALL of A, B, C, k_lateral, k_wave, wave_threshold_speed are placeholder
regression-style constants and are exposed as configurable fields with
"UNVALIDATED DEFAULT" docstrings -- see MeltRatesConfig below. Do not treat
the numeric defaults as scientifically validated; they exist so the module
runs out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from iceberg_model.config.physics_config import PhysicsConfig
from iceberg_model.state.iceberg_state import EnvironmentalState, IcebergState

SECONDS_PER_DAY = 86400.0


class MeltRatesConfig(BaseModel):
    """UNVALIDATED DEFAULTS. Calibrate against observations before
    quantitative use; see module docstring for the parameterization forms
    and their literature basis."""

    basal_A: float = Field(default=0.0, description="Basal melt regression intercept [m/day].")
    basal_B: float = Field(default=0.0034, description="Basal melt regression coefficient.")
    basal_C: float = Field(default=0.8, description="Basal melt relative-speed exponent.")
    lateral_k: float = Field(default=0.005, description="Lateral melt coefficient [m/day/C].")
    wave_k: float = Field(default=0.01, description="Wave erosion coefficient [m/day per m/s].")
    wave_threshold_speed_ms: float = Field(default=5.0, description="Wind speed threshold below which wave erosion is negligible [m/s].")
    iceberg_temp_c: float = Field(default=0.0, description="Assumed iceberg melting-interface temperature [C].")


@dataclass
class MeltRates:
    """All rates in [m/s], geometry-derivative-ready."""
    basal_rate: float
    lateral_rate: float
    wave_rate: float

    @property
    def dH_dt(self) -> float:
        return -self.basal_rate

    @property
    def dL_dt(self) -> float:
        return -(self.lateral_rate + self.wave_rate)

    @property
    def dW_dt(self) -> float:
        return -(self.lateral_rate + self.wave_rate)


def _m_per_day_to_m_per_s(x: float) -> float:
    return x / SECONDS_PER_DAY


def calculate_melt_rates(state: IcebergState, env: EnvironmentalState,
                          config: PhysicsConfig, melt_rates_config: MeltRatesConfig) -> MeltRates:
    m = config.melting
    basal = lateral = wave = 0.0

    T_w = env.sst
    have_sst = T_w is not None and env.validity_mask.get("sst", True)

    if m.basal_melt_enabled and have_sst and env.ocean_velocity is not None:
        U_rel = np.linalg.norm(env.ocean_velocity - state.velocity)
        dT = max(T_w - melt_rates_config.iceberg_temp_c, 0.0)
        M_basal_day = melt_rates_config.basal_A + melt_rates_config.basal_B * (U_rel ** melt_rates_config.basal_C) * dT
        basal = _m_per_day_to_m_per_s(max(M_basal_day, 0.0))

    if m.lateral_melt_enabled and have_sst:
        dT = max(T_w - melt_rates_config.iceberg_temp_c, 0.0)
        M_lat_day = melt_rates_config.lateral_k * dT
        lateral = _m_per_day_to_m_per_s(max(M_lat_day, 0.0))

    if m.wave_erosion_enabled and env.wind_velocity is not None:
        wind_speed = np.linalg.norm(env.wind_velocity)
        excess = max(wind_speed - melt_rates_config.wave_threshold_speed_ms, 0.0)
        M_wave_day = melt_rates_config.wave_k * excess
        wave = _m_per_day_to_m_per_s(M_wave_day)

    return MeltRates(basal_rate=basal, lateral_rate=lateral, wave_rate=wave)
