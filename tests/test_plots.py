import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from equity_pricing import (
    CalibrationResult,
    FlatMarketInputs,
    HestonParams,
    MarketSmile,
    MarketSurface,
    SmileQuote,
    calibrate_smile,
    plot_market_smile,
    plot_residual_heatmap,
    plot_smile_fit,
    plot_surface_fit,
)


@pytest.fixture
def sample_smile() -> MarketSmile:
    return MarketSmile(
        quotes=(
            SmileQuote(strike=90.0, maturity=1.0, implied_vol=0.27),
            SmileQuote(strike=100.0, maturity=1.0, implied_vol=0.24),
            SmileQuote(strike=110.0, maturity=1.0, implied_vol=0.22),
        )
    )


def test_plot_market_smile_returns_figure_and_axes(sample_smile: MarketSmile) -> None:
    figure, axes = plot_market_smile(sample_smile)

    assert figure.axes == [axes]
    assert axes.get_xlabel() == "Strike"
    assert axes.get_ylabel() == "Implied Volatility"
    assert axes.get_title() == "Market Smile (T=1.00)"

    plt.close(figure)


def test_plot_market_smile_uses_custom_title(sample_smile: MarketSmile) -> None:
    figure, axes = plot_market_smile(sample_smile, title="Test Smile")

    assert axes.get_title() == "Test Smile"

    plt.close(figure)


def test_plot_market_smile_draws_single_line(sample_smile: MarketSmile) -> None:
    figure, axes = plot_market_smile(sample_smile)
    line = axes.lines[0]

    assert len(axes.lines) == 1
    assert list(line.get_xdata()) == [90.0, 100.0, 110.0]
    assert list(line.get_ydata()) == [0.27, 0.24, 0.22]

    plt.close(figure)


@pytest.fixture
def sample_calibration_result(sample_smile: MarketSmile) -> CalibrationResult:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)
    initial_params = HestonParams(kappa=1.7, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05)
    return calibrate_smile(sample_smile, market, initial_params)


@pytest.fixture
def sample_surface() -> MarketSurface:
    return MarketSurface(
        smiles=(
            MarketSmile(
                quotes=(
                    SmileQuote(strike=90.0, maturity=0.75, implied_vol=0.25),
                    SmileQuote(strike=100.0, maturity=0.75, implied_vol=0.22),
                )
            ),
            MarketSmile(
                quotes=(
                    SmileQuote(strike=90.0, maturity=1.25, implied_vol=0.27),
                    SmileQuote(strike=100.0, maturity=1.25, implied_vol=0.24),
                    SmileQuote(strike=110.0, maturity=1.25, implied_vol=0.22),
                )
            ),
        )
    )


@pytest.fixture
def sample_surface_result(sample_surface: MarketSurface) -> CalibrationResult:
    market_vols = np.concatenate([smile.implied_vols for smile in sample_surface.smiles])
    model_vols = market_vols + np.array([0.005, -0.004, 0.003, -0.002, 0.001])
    residuals = model_vols - market_vols
    return CalibrationResult(
        params=HestonParams(kappa=1.7, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05),
        residuals=residuals,
        objective_value=0.5 * float(np.dot(residuals, residuals)),
        model_vols=model_vols,
        market_vols=market_vols,
        rmse=float(np.sqrt(np.mean(residuals * residuals))),
        mae=float(np.mean(np.abs(residuals))),
        max_abs_error=float(np.max(np.abs(residuals))),
        success=True,
        nfev=1,
        message="synthetic",
        n_restarts=1,
    )


def test_plot_smile_fit_returns_two_axes(
    sample_smile: MarketSmile,
    sample_calibration_result: CalibrationResult,
) -> None:
    figure, (smile_axes, residual_axes) = plot_smile_fit(sample_smile, sample_calibration_result)

    assert figure.axes == [smile_axes, residual_axes]
    assert smile_axes.get_ylabel() == "Implied Volatility"
    assert residual_axes.get_xlabel() == "Strike"
    assert residual_axes.get_ylabel() == "Residual"

    plt.close(figure)


def test_plot_smile_fit_uses_custom_title(
    sample_smile: MarketSmile,
    sample_calibration_result: CalibrationResult,
) -> None:
    figure, (smile_axes, _) = plot_smile_fit(
        sample_smile,
        sample_calibration_result,
        title="Calibration Fit",
    )

    assert smile_axes.get_title() == "Calibration Fit"

    plt.close(figure)


def test_plot_smile_fit_draws_market_model_and_residuals(
    sample_smile: MarketSmile,
    sample_calibration_result: CalibrationResult,
) -> None:
    figure, (smile_axes, residual_axes) = plot_smile_fit(sample_smile, sample_calibration_result)

    assert len(smile_axes.lines) == 2
    assert len(residual_axes.patches) == len(sample_smile.strikes)

    plt.close(figure)


def test_plot_surface_fit_returns_axes_per_expiry(
    sample_surface: MarketSurface,
    sample_surface_result: CalibrationResult,
) -> None:
    figure, axes = plot_surface_fit(sample_surface, sample_surface_result)

    assert len(axes) == len(sample_surface.smiles)
    assert axes[-1].get_xlabel() == "Strike"

    plt.close(figure)


def test_plot_surface_fit_uses_custom_title(
    sample_surface: MarketSurface,
    sample_surface_result: CalibrationResult,
) -> None:
    figure, _ = plot_surface_fit(sample_surface, sample_surface_result, title="Surface Fit")

    assert figure._suptitle.get_text() == "Surface Fit"

    plt.close(figure)


def test_plot_residual_heatmap_returns_figure_and_axes(
    sample_surface: MarketSurface,
    sample_surface_result: CalibrationResult,
) -> None:
    figure, axes = plot_residual_heatmap(sample_surface, sample_surface_result)

    assert figure.axes[0] is axes
    assert axes.get_xlabel() == "Strike"
    assert axes.get_ylabel() == "Maturity"
    assert axes.get_title() == "Residual Heatmap"

    plt.close(figure)
