"""
Numerical-convergence and general statistical metrics (spec principle F:
"numerical convergence must be testable").
"""

from __future__ import annotations

import numpy as np


def richardson_convergence_order(errors: list, step_sizes: list) -> float:
    """
    Estimate empirical convergence order p from two runs at different step
    sizes h1 > h2, assuming error ~ C * h^p:

        p = log(err1/err2) / log(h1/h2)

    Used in tests/test_solver.py to verify RK45/DOP853 achieve their
    expected order on a synthetic problem with a known analytic solution.
    """
    if len(errors) != 2 or len(step_sizes) != 2:
        raise ValueError("richardson_convergence_order expects exactly two (error, step) pairs.")
    e1, e2 = errors
    h1, h2 = step_sizes
    if e2 == 0 or h2 == 0:
        raise ValueError("Degenerate inputs (zero error or step size).")
    return float(np.log(e1 / e2) / np.log(h1 / h2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.abs(a - b)))
