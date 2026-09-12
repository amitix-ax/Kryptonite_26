"""
Central, typed configuration for the iceberg drift physics engine.

Per engineering principle G ("every physical parameter must be
configurable"), nothing in physics/* should hardcode a coefficient,
density, or toggle — everything is read from a `PhysicsConfig` instance
that call sites construct explicitly (with documented defaults) or load
from configs/default.yaml.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iceberg_model.physics import constants as C


class OceanDragMode(str, Enum):
    DEPTH_AVERAGED = "depth_averaged"
    DEPTH_INTEGRATED = "depth_integrated"


class SolverKind(str, Enum):
    RK45 = "RK45"
    DOP853 = "DOP853"
    RADAU = "Radau"


class FluidProperties(BaseModel):
    """Densities. Defaults are literature-typical, not iceberg-specific."""

    rho_seawater: float = Field(
        default=C.RHO_SEAWATER_DEFAULT, gt=0,
        description="Seawater density [kg/m^3].",
    )
    rho_air: float = Field(
        default=C.RHO_AIR_DEFAULT, gt=0,
        description="Air density [kg/m^3].",
    )
    rho_ice: float = Field(
        default=C.RHO_ICE_DEFAULT, gt=0,
        description="Iceberg (glacial ice) density [kg/m^3].",
    )

    @model_validator(mode="after")
    def _check_buoyancy_sanity(self) -> "FluidProperties":
        if not (0 < self.rho_ice < self.rho_seawater):
            raise ValueError(
                "Physical constraint violated: require 0 < rho_ice < "
                f"rho_seawater (got rho_ice={self.rho_ice}, "
                f"rho_seawater={self.rho_seawater}). A freely floating "
                "iceberg cannot be modeled otherwise."
            )
        return self


class DragCoefficients(BaseModel):
    c_dw: float = Field(default=C.C_DW_DEFAULT, gt=0, description="Ocean drag coefficient.")
    c_da: float = Field(default=C.C_DA_DEFAULT, gt=0, description="Wind drag coefficient.")
    c_dsi: float = Field(default=C.C_DSI_DEFAULT, gt=0, description="Sea-ice drag coefficient.")


class GroundingConfig(BaseModel):
    enabled: bool = True
    friction_coefficient_mu: float = Field(
        default=0.4, ge=0,
        description=(
            "UNVALIDATED DEFAULT. Coulomb-style friction coefficient for "
            "the seabed-keel interface. No single literature value exists; "
            "must be calibrated per region/sediment type."
        ),
    )
    normal_force_model: str = Field(
        default="weight_minus_buoyancy",
        description=(
            "How the normal force N is computed while grounded. Default "
            "assumes N = max(0, (m - rho_w*V_sub)*g), i.e. the portion of "
            "iceberg weight not supported by buoyancy. This is a simplified "
            "prototype model — sediment resistance, seabed slope, and "
            "partial grounding are NOT modeled yet (see docs/physics.md)."
        ),
    )


class MeltingConfig(BaseModel):
    enabled: bool = True
    basal_melt_enabled: bool = True
    lateral_melt_enabled: bool = True
    wave_erosion_enabled: bool = True
    min_dimension_m: float = Field(
        default=1.0, gt=0,
        description="Iceberg is terminated (melted out) once L, W, or H drop below this.",
    )


class PressureGradientConfig(BaseModel):
    enabled: bool = False
    reason_if_disabled: str = (
        "Disabled by default: requires sea-surface-height or density field "
        "which is frequently unavailable. Never fabricated when disabled."
    )


class SeaIceDragConfig(BaseModel):
    enabled: bool = True
    fallback_velocity_source: str = Field(
        default="ocean_current",
        description=(
            "If no explicit sea-ice velocity field is supplied, U_si falls "
            "back to this field (default: ocean surface current). This is "
            "an approximation, not an observation, and is logged as such "
            "in EnvironmentalState.validity_mask."
        ),
    )


class TemporalInterpolationConfig(BaseModel):
    allow_extrapolation: bool = False
    method: str = "linear"


class SolverConfig(BaseModel):
    kind: SolverKind = SolverKind.RK45
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step_seconds: Optional[float] = None


class CRSConfig(BaseModel):
    working_crs: str = Field(
        default="EPSG:3031",
        description="Projected CRS used internally for all dynamics (default: Antarctic Polar Stereographic).",
    )
    geographic_crs: str = Field(
        default="EPSG:4326",
        description="Geographic CRS used only for input/output/display.",
    )


class PhysicsConfig(BaseModel):
    """Top-level configuration object passed through the whole pipeline."""

    fluids: FluidProperties = Field(default_factory=FluidProperties)
    drag: DragCoefficients = Field(default_factory=DragCoefficients)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    melting: MeltingConfig = Field(default_factory=MeltingConfig)
    pressure_gradient: PressureGradientConfig = Field(default_factory=PressureGradientConfig)
    sea_ice_drag: SeaIceDragConfig = Field(default_factory=SeaIceDragConfig)
    temporal_interpolation: TemporalInterpolationConfig = Field(
        default_factory=TemporalInterpolationConfig
    )
    solver: SolverConfig = Field(default_factory=SolverConfig)
    crs: CRSConfig = Field(default_factory=CRSConfig)

    ocean_drag_mode: OceanDragMode = OceanDragMode.DEPTH_AVERAGED
    coriolis_enabled: bool = True

    model_config = ConfigDict(use_enum_values=False)
