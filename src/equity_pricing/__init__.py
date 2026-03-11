"""Equity pricing models and calibration utilities."""

from equity_pricing.black_scholes import (
    discount_factor,
    forward_price,
    price_bounds,
    price_european,
    vega,
)
from equity_pricing.calibration import (
    calibrate_surface,
    calibrate_smile,
    smile_objective_from_unconstrained,
    smile_residuals,
    surface_objective_from_unconstrained,
    surface_residuals,
)
from equity_pricing.heston import (
    heston_characteristic_function,
    heston_lewis_integrand,
    integrate_heston_integrand,
    model_smile,
    model_surface,
    price_european as price_european_heston,
)
from equity_pricing.implied_vol import implied_vol_from_price
from equity_pricing.plots import (
    plot_market_smile,
    plot_residual_heatmap,
    plot_smile_fit,
    plot_surface_fit,
)
from equity_pricing.simulation import (
    draw_correlated_normals,
    make_rng,
    make_time_grid,
    price_vanilla_mc,
    qe_variance_step,
    simulate_heston_paths,
)
from equity_pricing.types import (
    CalibrationSettings,
    CalibrationResult,
    FlatMarketInputs,
    HestonParams,
    MarketSmile,
    MarketSurface,
    MonteCarloResult,
    OptionSide,
    SmileQuote,
    VanillaOption,
)

__all__ = [
    "__version__",
    "CalibrationSettings",
    "CalibrationResult",
    "FlatMarketInputs",
    "HestonParams",
    "MarketSmile",
    "MarketSurface",
    "MonteCarloResult",
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
    "model_surface",
    "price_european_heston",
    "implied_vol_from_price",
    "calibrate_surface",
    "calibrate_smile",
    "smile_objective_from_unconstrained",
    "smile_residuals",
    "surface_objective_from_unconstrained",
    "surface_residuals",
    "plot_market_smile",
    "plot_residual_heatmap",
    "plot_smile_fit",
    "plot_surface_fit",
    "draw_correlated_normals",
    "make_rng",
    "make_time_grid",
    "price_vanilla_mc",
    "qe_variance_step",
    "simulate_heston_paths",
]

__version__ = "0.1.0"
