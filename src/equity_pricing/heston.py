"""Heston model characteristic-function utilities."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from equity_pricing.black_scholes import discount_factor
from equity_pricing.implied_vol import implied_vol_from_price
from equity_pricing.types import FlatMarketInputs, HestonParams, MarketSmile, MarketSurface, SmileQuote
from equity_pricing.types import OptionSide, VanillaOption


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

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        iu = 1j * argument
        beta = kappa - rho * sigma * iu
        d = _ensure_positive_real_part(
            np.sqrt(beta * beta + sigma * sigma * (iu + argument * argument))
        )
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value, error = quad(
            integrand,
            0.0,
            upper_limit,
            epsabs=abs_tol,
            epsrel=rel_tol,
            limit=limit,
        )
    return float(value), float(error)


def _heston_probability_integrand(
    u: float,
    strike: float,
    maturity: float,
    market: FlatMarketInputs,
    params: HestonParams,
    probability_index: int,
) -> float:
    if u == 0.0:
        return 0.0

    argument = complex(u, -1.0) if probability_index == 1 else complex(u, 0.0)
    numerator = np.exp(-1j * u * math.log(strike)) * heston_characteristic_function(
        argument,
        maturity,
        market,
        params,
    )
    if probability_index == 1:
        normalization = heston_characteristic_function(-1j, maturity, market, params)
        if not np.isfinite(normalization) or normalization == 0.0:
            return float("nan")
        numerator = numerator / normalization

    return float(np.real(numerator / (1j * u)))


def _price_call_scalar(
    strike: float,
    maturity: float,
    market: FlatMarketInputs,
    params: HestonParams,
    *,
    upper_limit: float,
    abs_tol: float,
    rel_tol: float,
    limit: int,
) -> float:
    discount_r = discount_factor(market.risk_free_rate, maturity)
    discount_q = discount_factor(market.dividend_yield, maturity)

    p1_integral, _ = integrate_heston_integrand(
        lambda u: _heston_probability_integrand(
            u,
            strike=strike,
            maturity=maturity,
            market=market,
            params=params,
            probability_index=1,
        ),
        upper_limit=upper_limit,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        limit=limit,
    )
    p2_integral, _ = integrate_heston_integrand(
        lambda u: _heston_probability_integrand(
            u,
            strike=strike,
            maturity=maturity,
            market=market,
            params=params,
            probability_index=2,
        ),
        upper_limit=upper_limit,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        limit=limit,
    )

    p1 = 0.5 + p1_integral / math.pi
    p2 = 0.5 + p2_integral / math.pi
    return max(0.0, market.spot * discount_q * p1 - strike * discount_r * p2)


def price_european(
    option: VanillaOption,
    market: FlatMarketInputs,
    params: HestonParams,
    *,
    upper_limit: float = 200.0,
    abs_tol: float = 1.0e-8,
    rel_tol: float = 1.0e-8,
    limit: int = 200,
) -> float | np.ndarray:
    """Price a European option under the Heston model via semi-analytic probabilities."""

    strikes = np.asarray(option.strike, dtype=float)
    scalar_input = strikes.ndim == 0
    strikes_1d = np.atleast_1d(strikes)

    discount_r = discount_factor(market.risk_free_rate, option.maturity)
    discount_q = discount_factor(market.dividend_yield, option.maturity)
    forward_intrinsic = market.spot * discount_q - strikes_1d * discount_r

    call_prices = np.array(
        [
            _price_call_scalar(
                float(strike),
                option.maturity,
                market,
                params,
                upper_limit=upper_limit,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                limit=limit,
            )
            for strike in strikes_1d
        ],
        dtype=float,
    )

    if option.side is OptionSide.CALL:
        prices = call_prices
    else:
        prices = call_prices - forward_intrinsic

    return float(prices[0]) if scalar_input else prices


def model_smile(
    strikes: np.ndarray,
    maturity: float,
    market: FlatMarketInputs,
    params: HestonParams,
    *,
    side: OptionSide = OptionSide.CALL,
    fill_value: float = np.nan,
    upper_limit: float = 200.0,
    abs_tol: float = 1.0e-8,
    rel_tol: float = 1.0e-8,
    limit: int = 200,
) -> np.ndarray:
    """Return model-implied vols across a strike grid for a single expiry."""

    strike_grid = np.asarray(strikes, dtype=float)
    if strike_grid.ndim != 1:
        raise ValueError(f"strikes must be a 1D array, got shape {strike_grid.shape!r}.")
    if np.any(strike_grid <= 0.0):
        raise ValueError("strikes must be positive.")
    if maturity <= 0.0:
        raise ValueError(f"maturity must be positive, got {maturity!r}.")

    implied_vols = np.empty_like(strike_grid, dtype=float)

    for index, strike in enumerate(strike_grid):
        option = VanillaOption(strike=float(strike), maturity=maturity, side=side)
        try:
            price = price_european(
                option,
                market,
                params,
                upper_limit=upper_limit,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                limit=limit,
            )
            implied_vols[index] = implied_vol_from_price(price, option, market)
        except (RuntimeError, ValueError):
            if np.isnan(fill_value):
                implied_vols[index] = np.nan
            else:
                raise

    return implied_vols


def model_surface(
    strikes_by_expiry: tuple[np.ndarray, ...],
    maturities: np.ndarray,
    market: FlatMarketInputs,
    params: HestonParams,
    *,
    side: OptionSide = OptionSide.CALL,
    fill_value: float = np.nan,
    upper_limit: float = 200.0,
    abs_tol: float = 1.0e-8,
    rel_tol: float = 1.0e-8,
    limit: int = 200,
) -> MarketSurface:
    """Return a Heston-implied volatility surface as ordered market-smile slices."""

    maturity_grid = np.asarray(maturities, dtype=float)
    if maturity_grid.ndim != 1:
        raise ValueError(f"maturities must be a 1D array, got shape {maturity_grid.shape!r}.")
    if len(strikes_by_expiry) != maturity_grid.size:
        raise ValueError("strikes_by_expiry and maturities must have the same length.")

    smiles: list[MarketSmile] = []
    for strikes, maturity in zip(strikes_by_expiry, maturity_grid, strict=True):
        implied_vols = model_smile(
            np.asarray(strikes, dtype=float),
            float(maturity),
            market,
            params,
            side=side,
            fill_value=fill_value,
            upper_limit=upper_limit,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            limit=limit,
        )
        quotes = tuple(
            SmileQuote(strike=float(strike), maturity=float(maturity), implied_vol=float(implied_vol))
            for strike, implied_vol in zip(np.asarray(strikes, dtype=float), implied_vols, strict=True)
        )
        smiles.append(MarketSmile(quotes))

    return MarketSurface(tuple(smiles))
