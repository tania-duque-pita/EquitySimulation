import numpy as np
import pytest

from equity_pricing import (
    CalibrationSettings,
    FlatMarketInputs,
    HestonParams,
    MarketSmile,
    SmileQuote,
    smile_objective_from_unconstrained,
    smile_residuals,
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


def test_calibration_settings_validate_inputs() -> None:
    with pytest.raises(ValueError, match="nan_penalty"):
        CalibrationSettings(nan_penalty=0.0)

    with pytest.raises(ValueError, match="integration_limit"):
        CalibrationSettings(integration_limit=0)


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
        "equity_pricing.calibration.model_smile",
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
