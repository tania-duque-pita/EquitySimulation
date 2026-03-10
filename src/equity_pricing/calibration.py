"""Calibration objective helpers for Heston smile fitting."""

from __future__ import annotations

import numpy as np

from equity_pricing.heston import model_smile
from equity_pricing.types import CalibrationSettings, FlatMarketInputs, HestonParams, MarketSmile


def smile_residuals(
    smile: MarketSmile,
    market: FlatMarketInputs,
    params: HestonParams,
    settings: CalibrationSettings | None = None,
) -> np.ndarray:
    """Return model-minus-market implied-vol residuals for a single smile."""

    calibration_settings = settings or CalibrationSettings()
    model_vols = model_smile(
        smile.strikes,
        smile.maturity,
        market,
        params,
        fill_value=np.nan,
        upper_limit=calibration_settings.upper_limit,
        abs_tol=calibration_settings.abs_tol,
        rel_tol=calibration_settings.rel_tol,
        limit=calibration_settings.integration_limit,
    )

    residuals = model_vols - smile.implied_vols
    return np.where(np.isnan(residuals), calibration_settings.nan_penalty, residuals)


def smile_objective_from_unconstrained(
    values: np.ndarray,
    smile: MarketSmile,
    market: FlatMarketInputs,
    settings: CalibrationSettings | None = None,
) -> np.ndarray:
    """Return smile residuals from an unconstrained Heston parameter vector."""

    params = HestonParams.from_unconstrained(values)
    return smile_residuals(smile, market, params, settings)
