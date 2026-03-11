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


def _clip_to_bounds(values: np.ndarray) -> HestonParams:
    clipped = np.array(
        [
            np.clip(values[0], *HestonParams.BOUNDS["kappa"]),
            np.clip(values[1], *HestonParams.BOUNDS["theta"]),
            np.clip(values[2], *HestonParams.BOUNDS["sigma"]),
            np.clip(values[3], *HestonParams.BOUNDS["rho"]),
            np.clip(values[4], *HestonParams.BOUNDS["v0"]),
        ],
        dtype=float,
    )
    return HestonParams(*clipped)


def _atm_variance(smile: MarketSmile, spot: float) -> float:
    atm_index = int(np.argmin(np.abs(smile.strikes - spot)))
    atm_vol = float(smile.implied_vols[atm_index])
    return atm_vol * atm_vol


def _domain_seed(
    target: MarketSmile | MarketSurface,
    market: FlatMarketInputs,
) -> HestonParams:
    if isinstance(target, MarketSmile):
        atm_variance = _atm_variance(target, market.spot)
    else:
        atm_variance = float(
            np.mean([_atm_variance(smile, market.spot) for smile in target.smiles])
        )

    return _clip_to_bounds(
        np.array([1.5, atm_variance, 0.5, -0.5, atm_variance], dtype=float)
    )


def _restart_vectors(
    target: MarketSmile | MarketSurface,
    market: FlatMarketInputs,
    initial_params: HestonParams,
    n_restarts: int,
) -> list[np.ndarray]:
    base = initial_params
    bumped = _clip_to_bounds(
        initial_params.as_array() * np.array([0.85, 1.15, 1.15, 1.0, 0.85], dtype=float)
    )
    domain = _domain_seed(target, market)

    seeds = [base, bumped, domain]
    restart_count = max(1, min(n_restarts, len(seeds)))

    unique_seeds: list[HestonParams] = []
    for seed in seeds:
        if not any(np.allclose(seed.as_array(), existing.as_array()) for existing in unique_seeds):
            unique_seeds.append(seed)

    return [seed.to_unconstrained() for seed in unique_seeds[:restart_count]]


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
    return _quote_residuals(smile, market, params, calibration_settings)


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


def _surface_model_vols(
    surface: MarketSurface,
    market: FlatMarketInputs,
    params: HestonParams,
    settings: CalibrationSettings,
) -> np.ndarray:
    return np.concatenate(
        [
            model_smile(
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
            for smile in surface.smiles
        ]
    )


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

    starts = _restart_vectors(smile, market, initial_params, calibration_settings.n_restarts)

    for start in starts:
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
        n_restarts=len(starts),
    )


def calibrate_surface(
    surface: MarketSurface,
    market: FlatMarketInputs,
    initial_params: HestonParams,
    settings: CalibrationSettings | None = None,
    expiry_weights: np.ndarray | None = None,
) -> CalibrationResult:
    """Calibrate Heston parameters to a full implied-volatility surface."""

    calibration_settings = settings or CalibrationSettings()
    best_result = None
    best_params = None
    best_residuals = None
    best_objective_value = np.inf
    total_nfev = 0

    starts = _restart_vectors(surface, market, initial_params, calibration_settings.n_restarts)

    for start in starts:
        result = least_squares(
            surface_objective_from_unconstrained,
            x0=start,
            args=(surface, market, calibration_settings, expiry_weights),
            method="trf",
        )
        total_nfev += int(result.nfev)

        candidate_params = HestonParams.from_unconstrained(result.x)
        candidate_residuals = surface_residuals(
            surface,
            market,
            candidate_params,
            calibration_settings,
            expiry_weights,
        )
        candidate_objective_value = 0.5 * float(np.dot(candidate_residuals, candidate_residuals))

        if candidate_objective_value < best_objective_value:
            best_result = result
            best_params = candidate_params
            best_residuals = candidate_residuals
            best_objective_value = candidate_objective_value

    assert best_result is not None
    assert best_params is not None
    assert best_residuals is not None

    model_vols = _surface_model_vols(surface, market, best_params, calibration_settings)
    market_vols = np.concatenate([smile.implied_vols for smile in surface.smiles])
    quote_residuals = model_vols - market_vols
    rmse = float(np.sqrt(np.mean(quote_residuals * quote_residuals)))
    mae = float(np.mean(np.abs(quote_residuals)))
    max_abs_error = float(np.max(np.abs(quote_residuals)))

    return CalibrationResult(
        params=best_params,
        residuals=best_residuals,
        objective_value=best_objective_value,
        model_vols=model_vols,
        market_vols=market_vols,
        rmse=rmse,
        mae=mae,
        max_abs_error=max_abs_error,
        success=bool(best_result.success),
        nfev=total_nfev,
        message=str(best_result.message),
        n_restarts=len(starts),
    )
