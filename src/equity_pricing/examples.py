"""End-to-end example workflows for the equity_pricing package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from equity_pricing.calibration import calibrate_smile, calibrate_surface
from equity_pricing.heston import model_surface, price_european
from equity_pricing.plots import plot_residual_heatmap, plot_smile_fit, plot_surface_fit
from equity_pricing.simulation import price_vanilla_mc
from equity_pricing.types import (
    CalibrationSettings,
    FlatMarketInputs,
    HestonParams,
    OptionSide,
    VanillaOption,
)


def run_end_to_end_example(
    *,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Build a synthetic surface, calibrate Heston, price an option, and plot diagnostics."""

    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    true_params = HestonParams(kappa=2.0, theta=0.04, sigma=0.2, rho=-0.3, v0=0.04)
    initial_params = HestonParams(kappa=1.4, theta=0.05, sigma=0.35, rho=-0.15, v0=0.05)

    maturities = np.array([0.5, 1.0], dtype=float)
    strikes_by_expiry = (
        np.array([90.0, 100.0, 110.0], dtype=float),
        np.array([85.0, 100.0, 115.0], dtype=float),
    )
    synthetic_surface = model_surface(
        strikes_by_expiry,
        maturities,
        market,
        true_params,
        upper_limit=120.0,
        abs_tol=1.0e-7,
        rel_tol=1.0e-7,
    )

    calibration_settings = CalibrationSettings(
        upper_limit=120.0,
        abs_tol=1.0e-7,
        rel_tol=1.0e-7,
        integration_limit=150,
        n_restarts=2,
    )
    surface_result = calibrate_surface(
        synthetic_surface,
        market,
        initial_params,
        calibration_settings,
    )
    smile_result = calibrate_smile(
        synthetic_surface.smiles[0],
        market,
        initial_params,
        calibration_settings,
    )

    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)
    analytic_price = price_european(
        option,
        market,
        surface_result.params,
        upper_limit=120.0,
        abs_tol=1.0e-7,
        rel_tol=1.0e-7,
        limit=150,
    )
    mc_result = price_vanilla_mc(
        option,
        market,
        surface_result.params,
        steps=32,
        n_paths=4_000,
        seed=1234,
    )

    smile_figure, _ = plot_smile_fit(
        synthetic_surface.smiles[0],
        smile_result,
        title="Synthetic Smile Fit",
    )
    surface_figure, _ = plot_surface_fit(
        synthetic_surface,
        surface_result,
        title="Synthetic Surface Fit",
    )
    heatmap_figure, _ = plot_residual_heatmap(
        synthetic_surface,
        surface_result,
        title="Synthetic Surface Residuals",
    )
    figures = {
        "smile_fit": smile_figure,
        "surface_fit": surface_figure,
        "residual_heatmap": heatmap_figure,
    }

    if save_dir is not None:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")

    if show:
        for figure in figures.values():
            figure.show()

    return {
        "market": market,
        "true_params": true_params,
        "initial_params": initial_params,
        "surface": synthetic_surface,
        "smile_result": smile_result,
        "surface_result": surface_result,
        "option": option,
        "analytic_price": float(analytic_price),
        "mc_result": mc_result,
        "figures": figures,
    }


def main() -> None:
    """Run the end-to-end example and print a brief summary."""

    results = run_end_to_end_example()
    surface_result = results["surface_result"]
    mc_result = results["mc_result"]
    print("Synthetic Heston calibration example")
    print(f"Calibrated params: {surface_result.params}")
    print(f"Surface RMSE: {surface_result.rmse:.6f}")
    print(f"Analytic option price: {results['analytic_price']:.6f}")
    print(
        "Monte Carlo price: "
        f"{mc_result.price:.6f} +/- {1.96 * mc_result.standard_error:.6f}"
    )


if __name__ == "__main__":
    main()
