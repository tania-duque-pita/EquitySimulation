"""Plotting helpers for market and model diagnostics."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from equity_pricing.types import MarketSmile


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
