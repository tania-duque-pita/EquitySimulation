"""Black-Scholes pricing helpers for European equity options."""

from __future__ import annotations

import math

import numpy as np

from equity_pricing.types import FlatMarketInputs, OptionSide, VanillaOption


def discount_factor(rate: float, maturity: float) -> float:
    """Return the continuously compounded discount factor."""

    return math.exp(-rate * maturity)


def forward_price(market: FlatMarketInputs, maturity: float) -> float:
    """Return the forward price under flat rates and dividend yield."""

    carry = market.risk_free_rate - market.dividend_yield
    return market.spot * math.exp(carry * maturity)


def _as_strike_array(strike: float | np.ndarray) -> np.ndarray:
    strikes = np.asarray(strike, dtype=float)
    if np.any(strikes <= 0.0):
        raise ValueError("strike must be positive.")
    return strikes


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values * values) / math.sqrt(2.0 * math.pi)


def price_bounds(
    option: VanillaOption,
    market: FlatMarketInputs,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return no-arbitrage lower and upper bounds for a European option price."""

    strikes = _as_strike_array(option.strike)
    discount = discount_factor(market.risk_free_rate, option.maturity)
    spot_discount = discount_factor(market.dividend_yield, option.maturity)
    discounted_spot = market.spot * spot_discount
    discounted_strike = strikes * discount

    if option.side is OptionSide.CALL:
        lower = np.maximum(discounted_spot - discounted_strike, 0.0)
        upper = np.full_like(strikes, discounted_spot)
    else:
        lower = np.maximum(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike

    if np.ndim(lower) == 0:
        return float(lower), float(upper)
    return lower, upper


def price_european(
    option: VanillaOption,
    market: FlatMarketInputs,
    vol: float,
) -> float | np.ndarray:
    """Price a European option under the Black-Scholes model."""

    if vol <= 0.0:
        raise ValueError(f"vol must be positive, got {vol!r}.")

    strikes = _as_strike_array(option.strike)
    sqrt_t = math.sqrt(option.maturity)
    total_vol = vol * sqrt_t
    forward = forward_price(market, option.maturity)
    discount = discount_factor(market.risk_free_rate, option.maturity)

    log_moneyness = np.log(forward / strikes)
    d1 = (log_moneyness + 0.5 * vol * vol * option.maturity) / total_vol
    d2 = d1 - total_vol

    if option.side is OptionSide.CALL:
        price = discount * (forward * _normal_cdf(d1) - strikes * _normal_cdf(d2))
    else:
        price = discount * (strikes * _normal_cdf(-d2) - forward * _normal_cdf(-d1))

    return float(price) if np.ndim(price) == 0 else price


def vega(
    option: VanillaOption,
    market: FlatMarketInputs,
    vol: float,
) -> float | np.ndarray:
    """Return Black-Scholes vega with respect to volatility."""

    if vol <= 0.0:
        raise ValueError(f"vol must be positive, got {vol!r}.")

    strikes = _as_strike_array(option.strike)
    sqrt_t = math.sqrt(option.maturity)
    total_vol = vol * sqrt_t
    forward = forward_price(market, option.maturity)
    discount = discount_factor(market.risk_free_rate, option.maturity)

    log_moneyness = np.log(forward / strikes)
    d1 = (log_moneyness + 0.5 * vol * vol * option.maturity) / total_vol
    result = discount * forward * sqrt_t * _normal_pdf(d1)

    return float(result) if np.ndim(result) == 0 else result
