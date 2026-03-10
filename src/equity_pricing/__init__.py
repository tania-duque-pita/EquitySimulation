"""Equity pricing models and calibration utilities."""

from equity_pricing.black_scholes import (
    discount_factor,
    forward_price,
    price_bounds,
    price_european,
    vega,
)
from equity_pricing.implied_vol import implied_vol_from_price
from equity_pricing.types import (
    FlatMarketInputs,
    MarketSmile,
    OptionSide,
    SmileQuote,
    VanillaOption,
)

__all__ = [
    "__version__",
    "FlatMarketInputs",
    "MarketSmile",
    "OptionSide",
    "SmileQuote",
    "VanillaOption",
    "discount_factor",
    "forward_price",
    "price_bounds",
    "price_european",
    "vega",
    "implied_vol_from_price",
]

__version__ = "0.1.0"
