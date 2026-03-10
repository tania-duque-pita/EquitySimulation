"""Black-Scholes implied-volatility inversion."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from equity_pricing.black_scholes import price_bounds, price_european
from equity_pricing.types import FlatMarketInputs, VanillaOption


def _validate_scalar_strike(option: VanillaOption) -> None:
    if np.ndim(option.strike) != 0:
        raise TypeError("implied_vol_from_price only supports scalar strikes.")


def implied_vol_from_price(
    price: float,
    option: VanillaOption,
    market: FlatMarketInputs,
    *,
    lower_vol: float = 1.0e-8,
    upper_vol: float = 1.0,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
) -> float:
    """Invert Black-Scholes implied volatility from a scalar option price."""

    _validate_scalar_strike(option)

    lower_price, upper_price = price_bounds(option, market)
    if not lower_price - tolerance <= price <= upper_price + tolerance:
        raise ValueError(
            "price must lie within no-arbitrage bounds "
            f"[{lower_price:.12f}, {upper_price:.12f}], got {price!r}."
        )

    def objective(vol: float) -> float:
        return price_european(option, market, vol) - price

    bracket_low = lower_vol
    bracket_high = upper_vol
    low_value = objective(bracket_low)

    if abs(low_value) <= tolerance or low_value > 0.0:
        return bracket_low

    while objective(bracket_high) < 0.0:
        bracket_high *= 2.0
        if bracket_high > 10.0:
            raise RuntimeError("failed to bracket implied volatility below 10.0.")

    return brentq(
        objective,
        bracket_low,
        bracket_high,
        xtol=tolerance,
        rtol=tolerance,
        maxiter=max_iterations,
    )
