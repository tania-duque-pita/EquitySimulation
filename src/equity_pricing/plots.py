"""Plotting helpers for market and model diagnostics."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from equity_pricing.types import CalibrationResult, MarketSmile, MarketSurface


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


def plot_volatility_surface(
    surface: MarketSurface,
    *,
    title: str | None = None,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """Plot a full implied-volatility surface in 3D."""

    strikes_by_smile = [smile.strikes for smile in surface.smiles]
    maturities = surface.maturities

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")

    same_strike_grid = all(
        np.array_equal(strikes_by_smile[0], strikes) for strikes in strikes_by_smile[1:]
    )

    if same_strike_grid:
        strike_grid = strikes_by_smile[0]
        strike_mesh, maturity_mesh = np.meshgrid(strike_grid, maturities)
        vol_mesh = np.array([smile.implied_vols for smile in surface.smiles], dtype=float)
        plotted = axes.plot_surface(
            strike_mesh,
            maturity_mesh,
            vol_mesh,
            cmap=cmap,
            linewidth=0.0,
            antialiased=True,
        )
    else:
        strikes = np.concatenate(strikes_by_smile)
        maturity_values = np.concatenate(
            [
                np.full(len(smile.quotes), smile.maturity, dtype=float)
                for smile in surface.smiles
            ]
        )
        vols = np.concatenate([smile.implied_vols for smile in surface.smiles])
        plotted = axes.plot_trisurf(
            strikes,
            maturity_values,
            vols,
            cmap=cmap,
            linewidth=0.2,
            antialiased=True,
        )

    axes.set_xlabel("Strike")
    axes.set_ylabel("Maturity")
    axes.set_zlabel("Implied Volatility")
    axes.set_title(title or "Implied Volatility Surface")
    figure.colorbar(plotted, ax=axes, shrink=0.7, pad=0.1, label="Implied Volatility")

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


def _surface_slices(surface: MarketSurface) -> list[slice]:
    slices: list[slice] = []
    start = 0
    for smile in surface.smiles:
        stop = start + len(smile.quotes)
        slices.append(slice(start, stop))
        start = stop
    return slices


def plot_surface_fit(
    surface: MarketSurface,
    result: CalibrationResult,
    *,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot market vs model smiles for each expiry in a calibrated surface."""

    axes_count = len(surface.smiles)
    figure, axes = plt.subplots(axes_count, 1, sharex=False, squeeze=False)
    flat_axes = axes[:, 0]
    slices = _surface_slices(surface)

    for smile, quote_slice, axis in zip(surface.smiles, slices, flat_axes, strict=True):
        axis.plot(smile.strikes, smile.implied_vols, marker="o", linestyle="-", label="Market")
        axis.plot(smile.strikes, result.model_vols[quote_slice], marker="s", linestyle="--", label="Model")
        axis.set_ylabel("Implied Vol")
        axis.set_title(f"T={smile.maturity:.2f}")
        axis.grid(True, alpha=0.3)

    flat_axes[0].legend()
    flat_axes[-1].set_xlabel("Strike")
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()

    return figure, flat_axes


def plot_residual_heatmap(
    surface: MarketSurface,
    result: CalibrationResult,
    *,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a maturity-strike heatmap of model-minus-market residuals."""

    slices = _surface_slices(surface)
    strikes = np.unique(
        np.concatenate([smile.strikes for smile in surface.smiles])
    )
    maturities = surface.maturities
    heatmap = np.full((len(surface.smiles), len(strikes)), np.nan, dtype=float)

    for row, (smile, quote_slice) in enumerate(zip(surface.smiles, slices, strict=True)):
        residuals = result.model_vols[quote_slice] - result.market_vols[quote_slice]
        for strike, residual in zip(smile.strikes, residuals, strict=True):
            col = int(np.where(strikes == strike)[0][0])
            heatmap[row, col] = residual

    figure, axes = plt.subplots()
    image = axes.imshow(heatmap, aspect="auto", interpolation="nearest")
    axes.set_xticks(np.arange(len(strikes)), labels=[f"{strike:.0f}" for strike in strikes])
    axes.set_yticks(np.arange(len(maturities)), labels=[f"{maturity:.2f}" for maturity in maturities])
    axes.set_xlabel("Strike")
    axes.set_ylabel("Maturity")
    axes.set_title(title or "Residual Heatmap")
    figure.colorbar(image, ax=axes, label="Residual")

    return figure, axes
