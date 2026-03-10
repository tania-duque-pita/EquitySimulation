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
    MarketSurface,
)


def _feller_penalty(params: HestonParams, weight: float) -> float:
    violation = max(0.0, params.sigma * params.sigma - 2.0 * params.kappa * params.theta)
    return weight * violation


def _restart_vectors(initial_params: HestonParams, n_restarts: int) -> list[np.ndarray]:
    base = initial_params.as_array()
    scales = [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([0.7, 1.2, 1.3, 0.8, 0.9]),
        np.array([1.4, 0.8, 0.7, 1.15, 1.1]),
        np.array([0.5, 1.5, 1.6, 0.6, 0.7]),
        np.array([1.8, 0.6, 0.5, 1.25, 1.3]),
        np.array([0.9, 0.9, 1.8, 1.1, 1.4]),
        np.array([1.1, 1.4, 0.6, 0.7, 0.8]),
        np.array([1.6, 1.1, 1.1, 1.3, 0.6]),
    ]
    bounded_vectors: list[np.ndarray] = []
    for scale in scales[:n_restarts]:
        candidate = base * scale
        clipped = np.array(
            [
                np.clip(candidate[0], *HestonParams.BOUNDS["kappa"]),
                np.clip(candidate[1], *HestonParams.BOUNDS["theta"]),
                np.clip(candidate[2], *HestonParams.BOUNDS["sigma"]),
                np.clip(candidate[3], *HestonParams.BOUNDS["rho"]),
                np.clip(candidate[4], *HestonParams.BOUNDS["v0"]),
            ],
            dtype=float,
        )
        bounded_vectors.append(HestonParams(*clipped).to_unconstrained())
    return bounded_vectors


def _quote_residuals(
    smile: MarketSmile,
    market: FlatMarketInputs,
    params: HestonParams,
    settings: CalibrationSettings,
) -> np.ndarray:
    model_vols = model_smile(
        smile.strikes,
        smile.maturity,
        market,
        params,
        fill_value=np.nan,
        upper_limit=settings.upper_limit,
        abs_tol=settings.abs_tol,
        rel_tol=settings.rel_tol,
        limit=settings.integration_limit,
    )
    return np.where(np.isnan(model_vols - smile.implied_vols), settings.nan_penalty, model_vols - smile.implied_vols)


def smile_residuals(
    smile: MarketSmile,
    market: FlatMarketInputs,
    params: HestonParams,
    settings: CalibrationSettings | None = None,
) -> np.ndarray:
    """Return model-minus-market implied-vol residuals for a single smile."""

    calibration_settings = settings or CalibrationSettings()
    residuals = _quote_residuals(smile, market, params, calibration_settings)

    if calibration_settings.enable_feller_penalty:
        residuals = np.concatenate(
            [residuals, np.array([_feller_penalty(params, calibration_settings.feller_penalty_weight)])]
        )
    return residuals


def smile_objective_from_unconstrained(
    values: np.ndarray,
    smile: MarketSmile,
    market: FlatMarketInputs,
    settings: CalibrationSettings | None = None,
) -> np.ndarray:
    """Return smile residuals from an unconstrained Heston parameter vector."""

    params = HestonParams.from_unconstrained(values)
    return smile_residuals(smile, market, params, settings)


def surface_residuals(
    surface: MarketSurface,
    market: FlatMarketInputs,
    params: HestonParams,
    settings: CalibrationSettings | None = None,
    expiry_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return stacked surface residuals with optional per-expiry weights."""

    calibration_settings = settings or CalibrationSettings()
    if expiry_weights is None:
        weights = np.ones(len(surface.smiles), dtype=float)
    else:
        weights = np.asarray(expiry_weights, dtype=float)
        if weights.shape != (len(surface.smiles),):
            raise ValueError(
                "expiry_weights must have shape "
                f"({len(surface.smiles)},), got {weights.shape!r}."
            )
        if np.any(weights <= 0.0):
            raise ValueError("expiry_weights must be positive.")

    residual_blocks = [
        weight * smile_residuals(smile, market, params, calibration_settings)
        for smile, weight in zip(surface.smiles, weights, strict=True)
    ]
    return np.concatenate(residual_blocks)


def surface_objective_from_unconstrained(
    values: np.ndarray,
    surface: MarketSurface,
    market: FlatMarketInputs,
    settings: CalibrationSettings | None = None,
    expiry_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return stacked surface residuals from an unconstrained parameter vector."""

    params = HestonParams.from_unconstrained(values)
    return surface_residuals(surface, market, params, settings, expiry_weights)


def calibrate_smile(
    smile: MarketSmile,
    market: FlatMarketInputs,
    initial_params: HestonParams,
    settings: CalibrationSettings | None = None,
) -> CalibrationResult:
    """Calibrate Heston parameters to a single market smile."""

    calibration_settings = settings or CalibrationSettings()
    best_result = None
    best_params = None
    best_residuals = None
    best_objective_value = np.inf
    total_nfev = 0

    for start in _restart_vectors(initial_params, calibration_settings.n_restarts):
        result = least_squares(
            smile_objective_from_unconstrained,
            x0=start,
            args=(smile, market, calibration_settings),
            method="trf",
        )
        total_nfev += int(result.nfev)

        candidate_params = HestonParams.from_unconstrained(result.x)
        candidate_residuals = smile_residuals(smile, market, candidate_params, calibration_settings)
        candidate_objective_value = 0.5 * float(np.dot(candidate_residuals, candidate_residuals))

        if candidate_objective_value < best_objective_value:
            best_result = result
            best_params = candidate_params
            best_residuals = candidate_residuals
            best_objective_value = candidate_objective_value

    assert best_result is not None
    assert best_params is not None
    assert best_residuals is not None
    model_vols = model_smile(
        smile.strikes,
        smile.maturity,
        market,
        best_params,
        fill_value=np.nan,
        upper_limit=calibration_settings.upper_limit,
        abs_tol=calibration_settings.abs_tol,
        rel_tol=calibration_settings.rel_tol,
        limit=calibration_settings.integration_limit,
    )
    quote_residuals = _quote_residuals(smile, market, best_params, calibration_settings)
    rmse = float(np.sqrt(np.mean(quote_residuals * quote_residuals)))
    mae = float(np.mean(np.abs(quote_residuals)))
    max_abs_error = float(np.max(np.abs(quote_residuals)))

    return CalibrationResult(
        params=best_params,
        residuals=best_residuals,
        objective_value=best_objective_value,
        model_vols=model_vols,
        market_vols=smile.implied_vols,
        rmse=rmse,
        mae=mae,
        max_abs_error=max_abs_error,
        success=bool(best_result.success),
        nfev=total_nfev,
        message=str(best_result.message),
        n_restarts=calibration_settings.n_restarts,
    )
