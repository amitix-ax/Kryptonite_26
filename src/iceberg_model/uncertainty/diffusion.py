"""
Diffusion matrix G(s_t, X_t) for the SDE extension (spec section 23).

    ds_t = f_phys(s_t, X_t) dt + G(s_t, X_t) dW_t

We do NOT assume G is known a priori. The initial supported form is
diagonal diffusion acting only on the velocity components (u, v) of the
7-element state -- representing unresolved environmental/model forcing,
not arbitrary noise on every state variable (position, geometry are left
deterministic at this stage; perturbing L/W/H stochastically without a
physical justification is explicitly avoided per spec).

Full (non-diagonal) covariance is a documented future extension point
(`DiffusionModel.full_covariance`), not implemented by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field


class DiffusionConfig(BaseModel):
    enabled: bool = False
    velocity_sigma_ms: float = Field(
        default=0.02,
        description=(
            "UNVALIDATED DEFAULT. Standard deviation [m/s] of the velocity "
            "diffusion term, representing unresolved forcing uncertainty. "
            "Must be calibrated against observed vs. physics-only residuals."
        ),
    )
    full_covariance: bool = Field(
        default=False,
        description="Placeholder for a future non-diagonal diffusion matrix. Not implemented.",
    )


def diagonal_velocity_diffusion_matrix(config: DiffusionConfig) -> np.ndarray:
    """
    Return the 7x7 diagonal diffusion matrix G, nonzero only on the (u, v)
    rows/cols (state indices 2, 3), per the "start with velocity residual
    uncertainty" guidance in the spec.
    """
    G = np.zeros((7, 7))
    if config.enabled:
        G[2, 2] = config.velocity_sigma_ms
        G[3, 3] = config.velocity_sigma_ms
    return G
