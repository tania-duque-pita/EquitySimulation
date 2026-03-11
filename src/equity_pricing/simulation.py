"""Monte Carlo scaffolding for Heston path simulation."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from equity_pricing.black_scholes import discount_factor
from equity_pricing.types import (
    FlatMarketInputs,
    HestonParams,
    MonteCarloResult,
    OptionSide,
    VanillaOption,
)


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


def simulate_heston_paths(
    market: FlatMarketInputs,
    params: HestonParams,
    maturity: float,
    steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = True,
    psi_threshold: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Heston spot and variance paths using QE variance updates."""

    if maturity <= 0.0:
        raise ValueError(f"maturity must be positive, got {maturity!r}.")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths!r}.")

    time_grid = make_time_grid(maturity, steps)
    dt = maturity / steps
    rng = make_rng(seed)

    effective_paths = n_paths
    base_paths = n_paths
    if antithetic:
        base_paths = (n_paths + 1) // 2
        effective_paths = 2 * base_paths

    z_var_base, z_spot_base = draw_correlated_normals(
        rng,
        rho=params.rho,
        steps=steps,
        n_paths=base_paths,
    )
    u_base = rng.uniform(size=(steps, base_paths))

    if antithetic:
        z_var = np.concatenate([z_var_base, -z_var_base], axis=1)[:, :effective_paths]
        z_spot = np.concatenate([z_spot_base, -z_spot_base], axis=1)[:, :effective_paths]
        u = np.concatenate([u_base, 1.0 - u_base], axis=1)[:, :effective_paths]
    else:
        z_var = z_var_base
        z_spot = z_spot_base
        u = u_base

    spot_paths = np.empty((steps + 1, effective_paths), dtype=float)
    variance_paths = np.empty((steps + 1, effective_paths), dtype=float)

    spot_paths[0] = market.spot
    variance_paths[0] = params.v0

    drift = market.risk_free_rate - market.dividend_yield
    gamma1 = 0.5
    gamma2 = 0.5
    rho2_complement = max(1.0 - params.rho * params.rho, 1.0e-16)
    k0 = -params.rho * params.kappa * params.theta * dt / params.sigma
    k1 = gamma1 * dt * (params.kappa * params.rho / params.sigma - 0.5) - (
        params.rho / params.sigma
    )
    k2 = gamma2 * dt * (params.kappa * params.rho / params.sigma - 0.5) + (
        params.rho / params.sigma
    )
    k3 = gamma1 * dt * rho2_complement
    k4 = gamma2 * dt * rho2_complement

    for index in range(steps):
        v_t = variance_paths[index]
        v_next = qe_variance_step(
            variance=v_t,
            dt=dt,
            params=params,
            normal_shocks=z_var[index],
            uniform_shocks=u[index],
            psi_threshold=psi_threshold,
        )
        variance_paths[index + 1] = v_next

        log_increment = (
            drift * dt
            + k0
            + k1 * v_t
            + k2 * v_next
            + np.sqrt(np.maximum(k3 * v_t + k4 * v_next, 0.0)) * z_spot[index]
        )
        spot_paths[index + 1] = spot_paths[index] * np.exp(log_increment)

    if antithetic:
        spot_paths = spot_paths[:, :n_paths]
        variance_paths = variance_paths[:, :n_paths]

    return time_grid, spot_paths, variance_paths


def price_vanilla_mc(
    option: VanillaOption,
    market: FlatMarketInputs,
    params: HestonParams,
    steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = True,
    psi_threshold: float = 1.5,
    confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Price a European vanilla option by Monte Carlo under Heston dynamics."""

    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError(
            f"confidence_level must lie strictly between 0 and 1, got {confidence_level!r}."
        )

    strike = np.asarray(option.strike, dtype=float)
    if strike.shape != ():
        raise ValueError("price_vanilla_mc requires a scalar strike.")

    _, spot_paths, _ = simulate_heston_paths(
        market=market,
        params=params,
        maturity=option.maturity,
        steps=steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
        psi_threshold=psi_threshold,
    )

    terminal_spots = spot_paths[-1]
    strike_value = float(strike)
    if option.side == OptionSide.CALL:
        payoffs = np.maximum(terminal_spots - strike_value, 0.0)
    else:
        payoffs = np.maximum(strike_value - terminal_spots, 0.0)

    discounted_payoffs = discount_factor(
        market.risk_free_rate,
        option.maturity,
    ) * payoffs
    price = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))

    z_score = NormalDist().inv_cdf(0.5 * (1.0 + confidence_level))
    half_width = z_score * standard_error
    confidence_interval = (price - half_width, price + half_width)

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        discounted_payoffs=discounted_payoffs,
        n_paths=n_paths,
    )
