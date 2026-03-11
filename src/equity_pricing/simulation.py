"""Monte Carlo scaffolding for Heston path simulation."""

from __future__ import annotations

import numpy as np


def make_time_grid(maturity: float, steps: int) -> np.ndarray:
    """Return an evenly spaced time grid including 0 and maturity."""

    if maturity <= 0.0:
        raise ValueError(f"maturity must be positive, got {maturity!r}.")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}.")

    return np.linspace(0.0, maturity, steps + 1, dtype=float)


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random number generator."""

    return np.random.default_rng(seed)


def draw_correlated_normals(
    rng: np.random.Generator,
    rho: float,
    steps: int,
    n_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw correlated standard normal shocks with shape (steps, n_paths)."""

    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie within [-1, 1], got {rho!r}.")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths!r}.")

    z1 = rng.standard_normal(size=(steps, n_paths))
    z2 = rng.standard_normal(size=(steps, n_paths))
    z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * z2
    return z1, z2
