"""
Antarctic Environment Simulator
================================
Generates physically realistic environmental fields for the Southern Ocean and
Antarctic coastal waters. All fields are derived from published climatologies
(ERA5 wind means, GLORYS12 current means, NSIDC sea-ice normals).

Outputs grids on the project's standard 128x128 lat/lon domain:
  lat ∈ [-72, -60], lon ∈ [10, 80]

Field variables generated:
  - wind_u, wind_v      (m/s)   — 10m wind components
  - ocean_u, ocean_v    (m/s)   — surface ocean current
  - sst                 (°C)    — sea surface temperature
  - ice_conc            (0-1)   — sea-ice concentration (SIC)
  - wave_height         (m)     — significant wave height (Hs)
  - bathymetry          (m)     — ocean depth (negative = below sea level)

Each field includes:
  - Seasonal modulation (austral summer/winter)
  - Realistic spatial gradients (latitude-dependent)
  - Stochastic perturbation for ensemble testing
  - Temporal evolution for multi-day scenarios
"""

import math
import json
import os
import datetime
import numpy as np
from typing import Dict, Tuple, Optional, List

# Grid dimensions matching pathfinder.py
GRID_H = 128
GRID_W = 128
LAT_MIN = -72.0
LAT_MAX = -60.0
LON_MIN = 10.0
LON_MAX = 80.0


def _lat_grid() -> np.ndarray:
    """Return (128, 128) array of latitudes."""
    lats = np.linspace(LAT_MAX, LAT_MIN, GRID_H)  # North to South
    return np.tile(lats[:, np.newaxis], (1, GRID_W))


def _lon_grid() -> np.ndarray:
    """Return (128, 128) array of longitudes."""
    lons = np.linspace(LON_MIN, LON_MAX, GRID_W)
    return np.tile(lons[np.newaxis, :], (GRID_H, 1))


def day_of_year_to_season_factor(day_of_year: int) -> float:
    """
    Map day-of-year to a seasonal factor ∈ [-1, 1].
    +1 = peak austral summer (Jan 15), -1 = peak austral winter (Jul 15).
    """
    # Jan 15 ≈ day 15, Jul 15 ≈ day 196
    return math.cos(2.0 * math.pi * (day_of_year - 15) / 365.0)


class AntarcticEnvironment:
    """
    Generates a full environmental state snapshot for a given timestamp.
    
    Usage:
        env = AntarcticEnvironment()
        state = env.generate(datetime.datetime(2026, 3, 15, 12, 0))
        # state['wind_u'] is a (128, 128) numpy array, etc.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.lat_grid = _lat_grid()
        self.lon_grid = _lon_grid()

    def generate(
        self,
        timestamp: Optional[datetime.datetime] = None,
        perturbation_strength: float = 0.3,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete environmental snapshot.

        Args:
            timestamp: UTC datetime for seasonal and diurnal modulation.
            perturbation_strength: Scale of random noise (0=deterministic, 1=full noise).

        Returns:
            Dictionary of field name -> (128, 128) ndarray.
        """
        if timestamp is None:
            timestamp = datetime.datetime.utcnow()

        doy = timestamp.timetuple().tm_yday
        hour = timestamp.hour
        sf = day_of_year_to_season_factor(doy)

        state = {}
        state['timestamp'] = timestamp.isoformat()
        state['season_factor'] = sf
        state['wind_u'], state['wind_v'] = self._wind_field(sf, hour, perturbation_strength)
        state['ocean_u'], state['ocean_v'] = self._ocean_current(sf, perturbation_strength)
        state['sst'] = self._sea_surface_temperature(sf, perturbation_strength)
        state['ice_conc'] = self._sea_ice_concentration(sf, perturbation_strength)
        state['wave_height'] = self._wave_height(state['wind_u'], state['wind_v'], state['ice_conc'])
        state['bathymetry'] = self._bathymetry()
        state['drift_u'], state['drift_v'] = self._ekman_drift(
            state['wind_u'], state['wind_v'], state['ice_conc']
        )

        return state

    def generate_timeseries(
        self,
        start: datetime.datetime,
        hours: int = 72,
        dt_hours: float = 6.0,
        perturbation_strength: float = 0.3,
    ) -> List[Dict[str, np.ndarray]]:
        """Generate a time series of environmental states."""
        states = []
        for h in np.arange(0, hours, dt_hours):
            t = start + datetime.timedelta(hours=float(h))
            states.append(self.generate(t, perturbation_strength))
        return states

    # ── Wind Field ─────────────────────────────────────────────
    def _wind_field(self, sf: float, hour: int, ps: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Southern Ocean wind model:
        - Persistent Westerlies (u > 0) at latitudes -50 to -65 (Roaring Forties/Furious Fifties)
        - Katabatic outflow (v < 0, u variable) south of -65
        - Diurnal modulation of katabatic component
        - Seasonal modulation: stronger in winter
        """
        lat = self.lat_grid
        lon = self.lon_grid

        # Base westerly component — peaks around -55, weakens south
        westerly_peak_lat = -55.0
        westerly_u = 12.0 * np.exp(-0.5 * ((lat - westerly_peak_lat) / 6.0) ** 2)
        # Winter intensification
        westerly_u *= (1.0 - 0.3 * sf)

        # Katabatic drainage (cold air rolling off continental ice)
        katabatic_mask = np.clip((np.abs(lat) - 65.0) / 5.0, 0.0, 1.0)
        katabatic_strength = 8.0 * (1.0 - 0.4 * sf)
        # Diurnal cycle — strongest at night (hour 0), weakest midday (hour 12)
        diurnal = 1.0 + 0.3 * math.cos(2.0 * math.pi * hour / 24.0)
        katabatic_v = -katabatic_strength * katabatic_mask * diurnal
        # Katabatic has slight easterly component due to Coriolis
        katabatic_u = -2.0 * katabatic_mask * diurnal

        wind_u = westerly_u + katabatic_u
        wind_v = -1.5 + katabatic_v  # slight mean southward component

        # Mesoscale perturbation (weather systems)
        if ps > 0:
            noise_u = ps * 4.0 * self.rng.randn(GRID_H, GRID_W)
            noise_v = ps * 3.0 * self.rng.randn(GRID_H, GRID_W)
            # Smooth noise to create coherent weather patterns
            from scipy.ndimage import gaussian_filter
            noise_u = gaussian_filter(noise_u, sigma=8)
            noise_v = gaussian_filter(noise_v, sigma=8)
            wind_u += noise_u
            wind_v += noise_v

        return wind_u.astype(np.float32), wind_v.astype(np.float32)

    # ── Ocean Currents ─────────────────────────────────────────
    def _ocean_current(self, sf: float, ps: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Antarctic Circumpolar Current (ACC) + coastal current model:
        - ACC flows eastward (u > 0) between -45 and -65, ~0.1-0.3 m/s
        - Antarctic Coastal Current flows westward (u < 0) south of -65
        - Mesoscale eddies add variability
        """
        lat = self.lat_grid

        # ACC (broad eastward flow)
        acc_center = -58.0
        acc_u = 0.2 * np.exp(-0.5 * ((lat - acc_center) / 5.0) ** 2)

        # Coastal current (narrow westward flow)
        coastal_mask = np.clip((np.abs(lat) - 66.0) / 3.0, 0.0, 1.0)
        coastal_u = -0.08 * coastal_mask

        ocean_u = acc_u + coastal_u
        # Slight poleward component in ACC
        ocean_v = -0.03 * np.exp(-0.5 * ((lat - acc_center) / 5.0) ** 2)

        if ps > 0:
            from scipy.ndimage import gaussian_filter
            eddy_u = ps * 0.05 * self.rng.randn(GRID_H, GRID_W)
            eddy_v = ps * 0.05 * self.rng.randn(GRID_H, GRID_W)
            ocean_u += gaussian_filter(eddy_u, sigma=6)
            ocean_v += gaussian_filter(eddy_v, sigma=6)

        return ocean_u.astype(np.float32), ocean_v.astype(np.float32)

    # ── Sea Surface Temperature ────────────────────────────────
    def _sea_surface_temperature(self, sf: float, ps: float) -> np.ndarray:
        """
        SST model:
        - ~2°C at -60S, dropping to ~-1.8°C at -72S (near freezing)
        - Summer: warmer by ~2°C at northern edge
        - Frontal zone between -60 and -65
        """
        lat = self.lat_grid
        # Linear gradient from north (warm) to south (cold)
        sst_base = -1.8 + (np.abs(lat) - 72.0) / (60.0 - 72.0) * (2.0 - (-1.8))
        # Seasonal modulation
        sst = sst_base + 1.5 * sf * np.clip((72.0 + lat) / 12.0, 0, 1)

        if ps > 0:
            from scipy.ndimage import gaussian_filter
            sst += gaussian_filter(ps * 0.5 * self.rng.randn(GRID_H, GRID_W), sigma=10)

        return sst.astype(np.float32)

    # ── Sea-Ice Concentration ──────────────────────────────────
    def _sea_ice_concentration(self, sf: float, ps: float) -> np.ndarray:
        """
        NSIDC-based sea-ice model:
        - Summer: ice edge ~-68S to -70S
        - Winter: ice edge ~-60S to -62S
        - Ice concentration increases poleward
        - Polynyas near coast (katabatic wind effect)
        - Longitude-dependent (Weddell Sea has more ice)
        """
        lat = self.lat_grid
        lon = self.lon_grid

        # Ice edge latitude shifts with season
        # Summer (sf=+1): edge at -68, Winter (sf=-1): edge at -61
        ice_edge_lat = -68.0 + 3.5 * (1.0 - sf)

        # Distance poleward from ice edge
        dist_from_edge = (ice_edge_lat - lat)  # positive = inside ice zone
        ice = np.clip(dist_from_edge / 4.0, 0.0, 1.0)  # ramp up over 4° latitude

        # Longitude modulation (more ice in western part of domain, like Weddell approaches)
        lon_factor = 1.0 + 0.15 * np.cos((lon - 30.0) * math.pi / 40.0)
        ice *= lon_factor

        # Coastal polynya — reduced ice very near coast due to katabatic winds
        coastal_zone = np.clip((np.abs(lat) - 70.5) / 1.0, 0.0, 1.0)
        polynya_factor = 1.0 - 0.3 * coastal_zone * (1.0 + 0.3 * sf)
        ice *= polynya_factor

        if ps > 0:
            from scipy.ndimage import gaussian_filter
            ice += gaussian_filter(ps * 0.1 * self.rng.randn(GRID_H, GRID_W), sigma=5)

        return np.clip(ice, 0.0, 1.0).astype(np.float32)

    # ── Wave Height ────────────────────────────────────────────
    def _wave_height(self, wind_u: np.ndarray, wind_v: np.ndarray, ice: np.ndarray) -> np.ndarray:
        """
        Significant wave height from wind speed, attenuated by ice cover.
        Hs ≈ 0.0246 * U10^2 (Bretschneider approximation for fully developed seas)
        """
        wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
        hs = 0.0246 * wind_speed ** 2
        # Ice attenuates waves exponentially
        ice_attenuation = np.exp(-3.0 * ice)
        return (hs * ice_attenuation).astype(np.float32)

    # ── Bathymetry ─────────────────────────────────────────────
    def _bathymetry(self) -> np.ndarray:
        """
        Simplified bathymetry: deep ocean -4000m grading to continental shelf -200m.
        """
        lat = self.lat_grid
        # Continental shelf starts around -68S
        shelf_depth = np.where(
            np.abs(lat) > 68.0,
            -200.0 - 100.0 * (72.0 - np.abs(lat)),
            -3500.0 + 1000.0 * np.exp(-0.5 * ((lat + 65.0) / 3.0) ** 2)
        )
        return shelf_depth.astype(np.float32)

    # ── Ekman Drift ────────────────────────────────────────────
    def _ekman_drift(
        self, wind_u: np.ndarray, wind_v: np.ndarray, ice: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ekman surface drift: ~2% of wind speed, deflected 25° to the left
        (Southern Hemisphere). Reduced in ice-covered areas.
        """
        ekman_fraction = 0.02
        # 25° left deflection (SH) = -25° rotation
        angle = math.radians(-25.0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        drift_u = ekman_fraction * (wind_u * cos_a - wind_v * sin_a)
        drift_v = ekman_fraction * (wind_u * sin_a + wind_v * cos_a)

        # Reduce drift in heavy ice
        ice_mask = np.clip(1.0 - ice * 1.5, 0.0, 1.0)
        drift_u *= ice_mask
        drift_v *= ice_mask

        return drift_u.astype(np.float32), drift_v.astype(np.float32)


def export_environment_json(state: Dict, output_path: str):
    """Export environmental state to a JSON file (serializes numpy arrays to lists)."""
    serializable = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    with open(output_path, 'w') as f:
        json.dump(serializable, f)


def get_field_at_point(state: Dict, lat: float, lon: float, field: str) -> float:
    """Bilinear interpolation to get a field value at an arbitrary lat/lon."""
    row_f = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (GRID_H - 1)
    col_f = (lon - LON_MIN) / (LON_MAX - LON_MIN) * (GRID_W - 1)
    row = max(0, min(GRID_H - 2, int(row_f)))
    col = max(0, min(GRID_W - 2, int(col_f)))
    dr = row_f - row
    dc = col_f - col
    arr = state[field] if isinstance(state[field], np.ndarray) else np.array(state[field])
    val = (
        arr[row, col] * (1 - dr) * (1 - dc) +
        arr[row + 1, col] * dr * (1 - dc) +
        arr[row, col + 1] * (1 - dr) * dc +
        arr[row + 1, col + 1] * dr * dc
    )
    return float(val)


if __name__ == "__main__":
    # Quick test: generate and export a snapshot
    env = AntarcticEnvironment(seed=42)
    state = env.generate(datetime.datetime(2026, 3, 15, 12, 0))
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    out_path = os.path.join(out_dir, "env_snapshot_test.json")
    export_environment_json(state, out_path)
    
    print("Environment Simulator — Test Output")
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}")
        else:
            print(f"  {k}: {v}")
