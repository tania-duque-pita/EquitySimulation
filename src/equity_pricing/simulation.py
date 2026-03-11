"""Monte Carlo scaffolding for Heston path simulation."""

from __future__ import annotations

import numpy as np

from equity_pricing.types import HestonParams


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


def qe_variance_step(
    variance: np.ndarray | float,
    dt: float,
    params: HestonParams,
    normal_shocks: np.ndarray | float,
    uniform_shocks: np.ndarray | float,
    psi_threshold: float = 1.5,
) -> np.ndarray:
    """Advance Heston variance one step using the Andersen QE scheme."""

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}.")
    if psi_threshold <= 1.0:
        raise ValueError(
            f"psi_threshold must be greater than 1.0, got {psi_threshold!r}."
        )

    v_t = np.asarray(variance, dtype=float)
    z = np.asarray(normal_shocks, dtype=float)
    u = np.asarray(uniform_shocks, dtype=float)

    if np.any(v_t < 0.0):
        raise ValueError("variance must be non-negative.")
    if z.shape != v_t.shape or u.shape != v_t.shape:
        raise ValueError(
            "variance, normal_shocks, and uniform_shocks must share the same shape."
        )
    if np.any((u <= 0.0) | (u >= 1.0)):
        raise ValueError("uniform_shocks must lie strictly between 0 and 1.")

    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma

    exp_kdt = np.exp(-kappa * dt)
    mean = theta + (v_t - theta) * exp_kdt
    variance_term = (
        v_t * sigma * sigma * exp_kdt * (1.0 - exp_kdt) / kappa
        + theta * sigma * sigma * (1.0 - exp_kdt) ** 2 / (2.0 * kappa)
    )
    psi = variance_term / np.maximum(mean * mean, 1.0e-16)

    next_variance = np.empty_like(mean)

    quadratic_mask = psi <= psi_threshold
    if np.any(quadratic_mask):
        psi_q = psi[quadratic_mask]
        z_q = z[quadratic_mask]
        mean_q = mean[quadratic_mask]
        two_over_psi = 2.0 / np.maximum(psi_q, 1.0e-16)
        b2 = two_over_psi - 1.0 + np.sqrt(two_over_psi) * np.sqrt(two_over_psi - 1.0)
        a = mean_q / (1.0 + b2)
        next_variance[quadratic_mask] = a * (np.sqrt(b2) + z_q) ** 2

    exponential_mask = ~quadratic_mask
    if np.any(exponential_mask):
        psi_e = psi[exponential_mask]
        mean_e = mean[exponential_mask]
        u_e = u[exponential_mask]
        p = (psi_e - 1.0) / (psi_e + 1.0)
        beta = (1.0 - p) / mean_e
        draws = np.log((1.0 - p) / (1.0 - u_e)) / beta
        next_variance[exponential_mask] = np.where(u_e <= p, 0.0, draws)

    return np.maximum(next_variance, 0.0)
