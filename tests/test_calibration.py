import numpy as np
import pytest

from equity_pricing.calibration import (
    _error_metrics,
    calibrate_surface,
    calibrate_smile,
    smile_objective_from_unconstrained,
    smile_residuals,
    surface_objective_from_unconstrained,
    surface_residuals,
)
from equity_pricing.heston import model_smile
from equity_pricing.types import (
    CalibrationResult,
    CalibrationSettings,
    FlatMarketInputs,
    HestonParams,
    MarketSmile,
    MarketSurface,
    SmileQuote,
)


@pytest.fixture
def sample_market() -> FlatMarketInputs:
    return FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)


@pytest.fixture
def sample_smile() -> MarketSmile:
    return MarketSmile(
        quotes=(
            SmileQuote(strike=95.0, maturity=1.25, implied_vol=0.21),
            SmileQuote(strike=100.0, maturity=1.25, implied_vol=0.20),
            SmileQuote(strike=105.0, maturity=1.25, implied_vol=0.19),
        )
    )


@pytest.fixture
def sample_params() -> HestonParams:
    return HestonParams(kappa=1.7, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05)


@pytest.fixture
def sample_surface() -> MarketSurface:
    return MarketSurface(
        smiles=(
            MarketSmile(
                quotes=(
                    SmileQuote(strike=95.0, maturity=0.75, implied_vol=0.22),
                    SmileQuote(strike=100.0, maturity=0.75, implied_vol=0.21),
                )
            ),
            MarketSmile(
                quotes=(
                    SmileQuote(strike=95.0, maturity=1.25, implied_vol=0.21),
                    SmileQuote(strike=100.0, maturity=1.25, implied_vol=0.20),
                    SmileQuote(strike=105.0, maturity=1.25, implied_vol=0.19),
                )
            ),
        )
    )


def test_calibration_settings_validate_inputs() -> None:
    with pytest.raises(ValueError, match="nan_penalty"):
        CalibrationSettings(nan_penalty=0.0)

    with pytest.raises(ValueError, match="integration_limit"):
        CalibrationSettings(integration_limit=0)

    with pytest.raises(ValueError, match="quadrature_points"):
        CalibrationSettings(quadrature_points=0)

    with pytest.raises(ValueError, match="n_restarts"):
        CalibrationSettings(n_restarts=0)


def test_smile_residuals_are_model_minus_market_vols(
    sample_smile: MarketSmile,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    residuals = smile_residuals(sample_smile, sample_market, sample_params)

    expected = np.array([-0.0066872, -0.00640884, -0.00529123])
    np.testing.assert_allclose(residuals, expected, rtol=1e-8, atol=1e-8)


def test_smile_residuals_use_default_equal_quote_weighting(
    sample_smile: MarketSmile,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    residuals = smile_residuals(sample_smile, sample_market, sample_params)

    assert residuals.shape == (3,)


def test_smile_residuals_use_nan_penalty_on_model_failure(
    sample_smile: MarketSmile,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "equity_pricing.calibration._model_smile_fast",
        lambda *args, **kwargs: np.array([0.2, np.nan, 0.18]),
    )

    residuals = smile_residuals(
        sample_smile,
        sample_market,
        sample_params,
        CalibrationSettings(nan_penalty=9.0),
    )

    np.testing.assert_allclose(residuals, np.array([-0.01, 9.0, -0.01]))


def test_smile_objective_from_unconstrained_matches_direct_residuals(
    sample_smile: MarketSmile,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    direct = smile_residuals(sample_smile, sample_market, sample_params)
    objective = smile_objective_from_unconstrained(
        sample_params.to_unconstrained(),
        sample_smile,
        sample_market,
    )

    np.testing.assert_allclose(objective, direct, rtol=1e-10, atol=1e-10)


def test_calibrate_smile_returns_structured_result(
    sample_smile: MarketSmile,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    result = calibrate_smile(sample_smile, sample_market, sample_params)

    assert isinstance(result, CalibrationResult)
    assert isinstance(result.params, HestonParams)
    assert result.residuals.shape == sample_smile.implied_vols.shape
    assert result.model_vols.shape == sample_smile.implied_vols.shape
    assert result.market_vols.shape == sample_smile.implied_vols.shape
    assert result.objective_value >= 0.0
    assert result.rmse >= 0.0
    assert result.mae >= 0.0
    assert result.max_abs_error >= 0.0
    assert result.nfev > 0
    assert isinstance(result.message, str)
    assert result.n_restarts == 3


def test_calibrate_smile_recovers_synthetic_parameters() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)
    true_params = HestonParams(kappa=1.6, theta=0.05, sigma=0.45, rho=-0.55, v0=0.045)
    strikes = np.array([85.0, 92.5, 100.0, 107.5, 115.0])
    target_vols = model_smile(strikes, 1.0, market, true_params)
    smile = MarketSmile(
        tuple(
            SmileQuote(strike=float(strike), maturity=1.0, implied_vol=float(vol))
            for strike, vol in zip(strikes, target_vols, strict=True)
        )
    )
    initial_params = HestonParams(kappa=1.2, theta=0.04, sigma=0.6, rho=-0.3, v0=0.04)

    result = calibrate_smile(smile, market, initial_params)

    assert result.success
    np.testing.assert_allclose(
        result.params.as_array(),
        true_params.as_array(),
        rtol=2.5e-1,
        atol=5.0e-3,
    )
    assert np.linalg.norm(result.residuals) < 1e-3
    assert result.rmse < 1e-3
    assert result.mae < 1e-3
    assert result.max_abs_error < 1e-3


def test_calibrate_smile_with_restarts_recovers_from_poor_initial_guess() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)
    true_params = HestonParams(kappa=1.6, theta=0.05, sigma=0.45, rho=-0.55, v0=0.045)
    strikes = np.array([85.0, 92.5, 100.0, 107.5, 115.0])
    target_vols = model_smile(strikes, 1.0, market, true_params)
    smile = MarketSmile(
        tuple(
            SmileQuote(strike=float(strike), maturity=1.0, implied_vol=float(vol))
            for strike, vol in zip(strikes, target_vols, strict=True)
        )
    )
    poor_initial_params = HestonParams(kappa=8.0, theta=0.2, sigma=2.0, rho=0.2, v0=0.2)

    result = calibrate_smile(
        smile,
        market,
        poor_initial_params,
        CalibrationSettings(n_restarts=3),
    )

    assert result.success
    assert result.n_restarts == 3
    assert np.linalg.norm(result.residuals) < 2e-3


def test_calibration_result_error_metrics_match_residuals(
    sample_smile: MarketSmile,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    result = calibrate_smile(sample_smile, sample_market, sample_params)

    expected_rmse = float(np.sqrt(np.mean(result.residuals * result.residuals)))
    expected_mae = float(np.mean(np.abs(result.residuals)))
    expected_max_abs_error = float(np.max(np.abs(result.residuals)))

    assert result.rmse == pytest.approx(expected_rmse)
    assert result.mae == pytest.approx(expected_mae)
    assert result.max_abs_error == pytest.approx(expected_max_abs_error)


def test_error_metrics_ignore_nan_residuals() -> None:
    rmse, mae, max_abs_error = _error_metrics(np.array([0.1, np.nan, -0.2]))

    assert rmse == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2.0))
    assert mae == pytest.approx(0.15)
    assert max_abs_error == pytest.approx(0.2)


def test_surface_residuals_stack_expiry_blocks_in_surface_order(
    sample_surface: MarketSurface,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    residuals = surface_residuals(sample_surface, sample_market, sample_params)

    assert residuals.shape == (5,)
    expected_first = smile_residuals(sample_surface.smiles[0], sample_market, sample_params)
    expected_second = smile_residuals(sample_surface.smiles[1], sample_market, sample_params)
    np.testing.assert_allclose(residuals[:2], expected_first, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(residuals[2:], expected_second, rtol=1e-10, atol=1e-10)


def test_surface_residuals_apply_optional_expiry_weights(
    sample_surface: MarketSurface,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    weights = np.array([2.0, 0.5])
    residuals = surface_residuals(sample_surface, sample_market, sample_params, expiry_weights=weights)

    expected_first = 2.0 * smile_residuals(sample_surface.smiles[0], sample_market, sample_params)
    expected_second = 0.5 * smile_residuals(sample_surface.smiles[1], sample_market, sample_params)
    np.testing.assert_allclose(residuals[:2], expected_first, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(residuals[2:], expected_second, rtol=1e-10, atol=1e-10)


def test_surface_residuals_reject_invalid_expiry_weights(
    sample_surface: MarketSurface,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        surface_residuals(
            sample_surface,
            sample_market,
            sample_params,
            expiry_weights=np.array([1.0]),
        )

    with pytest.raises(ValueError, match="positive"):
        surface_residuals(
            sample_surface,
            sample_market,
            sample_params,
            expiry_weights=np.array([1.0, 0.0]),
        )


def test_surface_objective_from_unconstrained_matches_direct_surface_residuals(
    sample_surface: MarketSurface,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    direct = surface_residuals(sample_surface, sample_market, sample_params)
    objective = surface_objective_from_unconstrained(
        sample_params.to_unconstrained(),
        sample_surface,
        sample_market,
    )

    np.testing.assert_allclose(objective, direct, rtol=1e-10, atol=1e-10)


def test_calibrate_surface_returns_structured_result(
    sample_surface: MarketSurface,
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    result = calibrate_surface(sample_surface, sample_market, sample_params)

    assert isinstance(result, CalibrationResult)
    assert result.model_vols.shape == (5,)
    assert result.market_vols.shape == (5,)
    assert result.objective_value >= 0.0
    assert result.rmse >= 0.0
    assert result.n_restarts == 3


def test_calibrate_surface_recovers_synthetic_parameters() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)
    true_params = HestonParams(kappa=1.5, theta=0.045, sigma=0.5, rho=-0.5, v0=0.05)
    strikes_by_expiry = (
        np.array([90.0, 100.0, 110.0]),
        np.array([85.0, 100.0, 115.0]),
    )
    maturities = np.array([0.75, 1.5])
    smiles = []
    for strikes, maturity in zip(strikes_by_expiry, maturities, strict=True):
        vols = model_smile(strikes, maturity, market, true_params)
        smiles.append(
            MarketSmile(
                tuple(
                    SmileQuote(strike=float(strike), maturity=float(maturity), implied_vol=float(vol))
                    for strike, vol in zip(strikes, vols, strict=True)
                )
            )
        )
    surface = MarketSurface(tuple(smiles))
    initial_params = HestonParams(kappa=0.9, theta=0.03, sigma=0.8, rho=-0.2, v0=0.03)

    result = calibrate_surface(surface, market, initial_params)

    assert result.success
    np.testing.assert_allclose(
        result.params.as_array(),
        true_params.as_array(),
        rtol=3.0e-1,
        atol=7.5e-3,
    )
    assert result.rmse < 1e-3
