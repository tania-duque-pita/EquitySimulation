"""Calibration objective helpers for Heston smile fitting."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from equity_pricing.heston import model_smile
from equity_pricing.types import (
    CalibrationResult,
    CalibrationSettings,
    FlatMarketInputs,
    HestonParams,
    MarketSmile,
)


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


def calibrate_smile(
    smile: MarketSmile,
    market: FlatMarketInputs,
    initial_params: HestonParams,
    settings: CalibrationSettings | None = None,
) -> CalibrationResult:
    """Calibrate Heston parameters to a single market smile."""

    calibration_settings = settings or CalibrationSettings()
    result = least_squares(
        smile_objective_from_unconstrained,
        x0=initial_params.to_unconstrained(),
        args=(smile, market, calibration_settings),
        method="trf",
    )

    calibrated_params = HestonParams.from_unconstrained(result.x)
    residuals = smile_residuals(smile, market, calibrated_params, calibration_settings)
    objective_value = 0.5 * float(np.dot(residuals, residuals))

    return CalibrationResult(
        params=calibrated_params,
        residuals=residuals,
        objective_value=objective_value,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
    )
