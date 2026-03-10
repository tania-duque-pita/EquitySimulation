"""Heston model characteristic-function utilities."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from equity_pricing.types import FlatMarketInputs, HestonParams


def _ensure_positive_real_part(values: np.ndarray) -> np.ndarray:
    return np.where(np.real(values) < 0.0, -values, values)


def heston_characteristic_function(
    u: complex | np.ndarray,
    maturity: float,
    market: FlatMarketInputs,
    params: HestonParams,
) -> complex | np.ndarray:
    """Return the Heston characteristic function for log spot at maturity."""

    if maturity <= 0.0:
        raise ValueError(f"maturity must be positive, got {maturity!r}.")

    argument = np.asarray(u, dtype=np.complex128)
    scalar_input = argument.ndim == 0

    x0 = np.log(market.spot)
    drift = market.risk_free_rate - market.dividend_yield
    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma
    rho = params.rho
    v0 = params.v0

    iu = 1j * argument
    beta = kappa - rho * sigma * iu
    d = _ensure_positive_real_part(np.sqrt(beta * beta + sigma * sigma * (iu + argument * argument)))
    g = (beta - d) / (beta + d)
    exp_neg_d_t = np.exp(-d * maturity)

    one_minus_g_exp = 1.0 - g * exp_neg_d_t
    one_minus_g = 1.0 - g

    c_term = (
        iu * (x0 + drift * maturity)
        + (kappa * theta / (sigma * sigma))
        * ((beta - d) * maturity - 2.0 * np.log(one_minus_g_exp / one_minus_g))
    )
    d_term = ((beta - d) / (sigma * sigma)) * ((1.0 - exp_neg_d_t) / one_minus_g_exp)
    values = np.exp(c_term + d_term * v0)

    return complex(values) if scalar_input else values


def heston_lewis_integrand(
    u: float | np.ndarray,
    log_moneyness: float,
    maturity: float,
    market: FlatMarketInputs,
    params: HestonParams,
) -> float | np.ndarray:
    """Return the damped Lewis-style Heston pricing integrand."""

    grid = np.asarray(u, dtype=float)
    if np.any(grid < 0.0):
        raise ValueError("integration variable u must be non-negative.")

    shifted_argument = grid - 0.5j
    characteristic_values = heston_characteristic_function(
        shifted_argument,
        maturity,
        market,
        params,
    )
    phase = np.exp(-1j * grid * log_moneyness)
    values = np.real(phase * characteristic_values / (grid * grid + 0.25))

    return float(values) if np.ndim(values) == 0 else values


def integrate_heston_integrand(
    integrand,
    *,
    upper_limit: float = 200.0,
    abs_tol: float = 1.0e-8,
    rel_tol: float = 1.0e-8,
    limit: int = 200,
) -> tuple[float, float]:
    """Numerically integrate a Heston-style scalar integrand on [0, upper_limit]."""

    if upper_limit <= 0.0:
        raise ValueError(f"upper_limit must be positive, got {upper_limit!r}.")

    value, error = quad(
        integrand,
        0.0,
        upper_limit,
        epsabs=abs_tol,
        epsrel=rel_tol,
        limit=limit,
    )
    return float(value), float(error)
