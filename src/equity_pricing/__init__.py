"""Equity pricing models and calibration utilities."""

from equity_pricing.black_scholes import discount_factor, forward_price, price_european
from equity_pricing.types import FlatMarketInputs, OptionSide, VanillaOption

__all__ = [
    "__version__",
    "FlatMarketInputs",
    "OptionSide",
    "VanillaOption",
    "discount_factor",
    "forward_price",
    "price_european",
]

__version__ = "0.1.0"
