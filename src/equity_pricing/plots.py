"""Plotting helpers for market and model diagnostics."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from equity_pricing.types import CalibrationResult, MarketSmile


def plot_market_smile(
    smile: MarketSmile,
    *,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot strike vs implied volatility for a single market smile."""

    figure, axes = plt.subplots()
    axes.plot(smile.strikes, smile.implied_vols, marker="o", linestyle="-")
    axes.set_xlabel("Strike")
    axes.set_ylabel("Implied Volatility")
    axes.set_title(title or f"Market Smile (T={smile.maturity:.2f})")
    axes.grid(True, alpha=0.3)

    return figure, axes


def plot_smile_fit(
    smile: MarketSmile,
    result: CalibrationResult,
    *,
    title: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot market vs model smile and quote residuals for a calibration result."""

    figure, (smile_axes, residual_axes) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    smile_axes.plot(smile.strikes, smile.implied_vols, marker="o", linestyle="-", label="Market")
    smile_axes.plot(smile.strikes, result.model_vols, marker="s", linestyle="--", label="Model")
    smile_axes.set_ylabel("Implied Volatility")
    smile_axes.set_title(title or f"Smile Fit (T={smile.maturity:.2f})")
    smile_axes.grid(True, alpha=0.3)
    smile_axes.legend()

    residual_axes.bar(smile.strikes, result.model_vols - result.market_vols, width=2.5)
    residual_axes.axhline(0.0, color="black", linewidth=1.0)
    residual_axes.set_xlabel("Strike")
    residual_axes.set_ylabel("Residual")
    residual_axes.grid(True, axis="y", alpha=0.3)

    return figure, (smile_axes, residual_axes)
