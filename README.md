# Equity Pricing

Incremental Python build-out of a Heston calibration and pricing framework for European equity options.

## Scope

The repository now covers the full workflow:

- Black-Scholes pricing and implied-vol inversion
- Heston semi-analytic pricing and implied-vol smile generation
- Smile and surface calibration
- Heston Monte Carlo simulation with an Andersen QE variance scheme
- Diagnostics, plots, and regression tests

## Package Layout

```text
src/equity_pricing/
tests/
```

Main modules:

- `black_scholes.py`: Black-Scholes pricing, vega, and no-arbitrage bounds
- `implied_vol.py`: implied-vol inversion from option prices
- `heston.py`: characteristic function pricing and model smile/surface generation
- `calibration.py`: smile and surface calibration objectives plus optimizers
- `simulation.py`: QE variance stepping, Heston path simulation, and Monte Carlo pricing
- `plots.py`: smile/surface fit plots and residual diagnostics
- `examples.py`: end-to-end synthetic workflow

## Local Setup

This repo works well with a local virtual environment. On your machine, `py` is the reliable launcher, so prefer it over the global `python` command.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pytest -p no:cacheprovider
```

If Matplotlib complains about config or cache directories in PowerShell, set:

```powershell
$env:TEMP="$PWD\.tmp"
$env:TMP=$env:TEMP
$env:MPLCONFIGDIR="$PWD\.tmp\mpl"
```

## Quick Start

Import the package:

```python
from equity_pricing import (
    CalibrationSettings,
    FlatMarketInputs,
    HestonParams,
    OptionSide,
    VanillaOption,
    implied_vol_from_price,
    model_smile,
    price_european,
    price_european_heston,
    price_vanilla_mc,
)
```

Black-Scholes example:

```python
from equity_pricing import FlatMarketInputs, OptionSide, VanillaOption, price_european

market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

price = price_european(option, market, vol=0.20)
print(price)
```

Heston semi-analytic example:

```python
from equity_pricing import (
    FlatMarketInputs,
    HestonParams,
    OptionSide,
    VanillaOption,
    price_european_heston,
)

market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
params = HestonParams(kappa=2.0, theta=0.04, sigma=0.2, rho=-0.3, v0=0.04)
option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

price = price_european_heston(option, market, params)
print(price)
```

Monte Carlo example:

```python
from equity_pricing import (
    FlatMarketInputs,
    HestonParams,
    OptionSide,
    VanillaOption,
    price_vanilla_mc,
)

market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
params = HestonParams(kappa=2.0, theta=0.04, sigma=0.2, rho=-0.3, v0=0.04)
option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

result = price_vanilla_mc(option, market, params, steps=64, n_paths=10_000, seed=123)
print(result.price, result.standard_error, result.confidence_interval)
```

## End-to-End Example

Run the synthetic calibration and diagnostics workflow:

```powershell
python -m equity_pricing.examples
```

From Python:

```python
from equity_pricing import run_end_to_end_example

results = run_end_to_end_example(save_dir="example_output")
print(results["surface_result"].params)
print(results["analytic_price"])
```

The example:

- generates a synthetic Heston surface
- calibrates a smile and a full surface
- prices a sample option analytically and by Monte Carlo
- creates smile-fit, surface-fit, and residual-heatmap figures

## Modeling Assumptions

This framework intentionally stays narrow and explicit:

- European vanilla equity options only
- flat risk-free rate `r` and flat dividend yield `q`
- Black-Scholes implied vols as the calibration target
- Heston risk-neutral dynamics with parameters `(kappa, theta, sigma, rho, v0)`
- semi-analytic pricing through the Heston characteristic function
- Monte Carlo variance simulation through the Andersen QE scheme

What is not in scope:

- American, barrier, Asian, or path-dependent pricing APIs
- stochastic interest rates or term structures
- bid/ask-aware calibration weights
- local volatility or jump-diffusion extensions

## Numerical Notes

- Black-Scholes implied vols are inverted with Brent root finding.
- Heston prices are computed by numerical integration of the characteristic-function probabilities.
- Smile and surface calibration use `scipy.optimize.least_squares` on implied-vol residuals.
- Monte Carlo paths use antithetic sampling by default.
- The Monte Carlo regression tests currently show the cleanest agreement with the semi-analytic engine in milder call-price configurations. Rougher parameter sets still deserve more numerical work if production-quality MC accuracy is required.

## Testing

The test suite includes:

- unit tests for pricing, vega, bounds, implied-vol inversion, and parameter validation
- regression tests for Heston characteristic functions and semi-analytic prices
- synthetic smile and surface calibration recovery tests
- plot smoke tests
- Monte Carlo path and pricing tests
- Monte Carlo vs semi-analytic consistency checks
- an end-to-end example smoke test

Run everything:

```powershell
python -m pytest -p no:cacheprovider
```

Run a focused file:

```powershell
python -m pytest tests/test_simulation.py -p no:cacheprovider
```

## Release Checks

For a clean local verification pass, use:

```powershell
python -m pytest -p no:cacheprovider
python -m ruff check src tests
```

If you want the exact Windows environment used during development in this repo:

```powershell
$env:TEMP="$PWD\.tmp"
$env:TMP=$env:TEMP
$env:MPLCONFIGDIR="$PWD\.tmp\mpl"
python -m pytest -p no:cacheprovider
python -m ruff check src tests
```

## Status

The 30-commit roadmap is complete through:

- pricing
- implied-vol inversion
- smile and surface calibration
- Monte Carlo simulation and pricing
- diagnostics and example workflows
