"""Equity pricing models and calibration utilities."""

from equity_pricing.black_scholes import (
    discount_factor,
    forward_price,
    price_bounds,
    price_european,
    vega,
)
from equity_pricing.calibration import smile_objective_from_unconstrained, smile_residuals
from equity_pricing.heston import (
    heston_characteristic_function,
    heston_lewis_integrand,
    integrate_heston_integrand,
    model_smile,
    price_european as price_european_heston,
)
from equity_pricing.implied_vol import implied_vol_from_price
from equity_pricing.plots import plot_market_smile
from equity_pricing.types import (
    CalibrationSettings,
    FlatMarketInputs,
    HestonParams,
    MarketSmile,
    OptionSide,
    SmileQuote,
    VanillaOption,
)

__all__ = [
    "__version__",
    "CalibrationSettings",
    "FlatMarketInputs",
    "HestonParams",
    "MarketSmile",
    "OptionSide",
    "SmileQuote",
    "VanillaOption",
    "discount_factor",
    "forward_price",
    "price_bounds",
    "price_european",
    "vega",
    "heston_characteristic_function",
    "heston_lewis_integrand",
    "integrate_heston_integrand",
    "model_smile",
    "price_european_heston",
    "implied_vol_from_price",
    "smile_objective_from_unconstrained",
    "smile_residuals",
    "plot_market_smile",
]

__version__ = "0.1.0"
