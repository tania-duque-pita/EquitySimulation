# Equity Pricing

Python project focused on equity-option pricing and Heston-model calibration.

This repository is positioned as a compact quantitative development / quantitative modelling portfolio project. It demonstrates model implementation, numerical methods, calibration workflows, testing discipline, and notebook-based communication of results.

## Repo Objective

Build a clean, testable library for European equity-option analytics with an emphasis on:

- Black-Scholes pricing and implied-volatility inversion
- Heston semi-analytic pricing
- Heston smile and surface calibration to market implied vols
- Heston Monte Carlo simulation with Andersen QE variance dynamics
- Diagnostics and visualization for calibration outputs

## Repo Current Capabilities

- Black-Scholes
  - European call/put pricing
  - no-arbitrage price bounds
  - vega
  - implied-volatility inversion

- Heston
  - characteristic-function implementation
  - semi-analytic European pricing
  - model smile generation
  - model surface generation

- Calibration
  - smile calibration
  - surface calibration
  - bounded parameter transforms
  - multiple restart seeds
  - calibration error reporting (`RMSE`, `MAE`, `MaxAbsError`)

- Simulation
  - Andersen QE variance stepping
  - correlated shock generation
  - Heston path simulation
  - Monte Carlo vanilla pricing with confidence intervals

- Engineering quality
  - unit and regression tests across pricing, calibration, simulation, and plotting
  - notebook demo for synthetic and market-data workflows

## Project Structure

```text
src/equity_pricing/
tests/
examples/examples.ipynb
```

Main modules:

- `black_scholes.py`: Black-Scholes pricing utilities
- `implied_vol.py`: implied-volatility inversion
- `heston.py`: Heston characteristic-function pricing and model smile/surface generation
- `calibration.py`: smile and surface calibration
- `simulation.py`: Monte Carlo and QE variance dynamics
- `plots.py`: calibration diagnostics and surface plots
- `types.py`: domain datatypes and settings

## Setup

The project is set up for local development with a virtual environment.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If you want to run the test suite:

```powershell
python -m pytest -p no:cacheprovider
```

If Matplotlib or temporary directories cause issues on Windows / PowerShell:

```powershell
$env:TEMP="$PWD\.tmp"
$env:TMP=$env:TEMP
$env:MPLCONFIGDIR="$PWD\.tmp\mpl"
```

## Demo

For the project walkthrough, go directly to:

[examples.ipynb](c:/Github_Repos/EquityPricing/examples/examples.ipynb)

The notebook is the intended demo surface for this repository. It shows how to:

- load / prepare implied-volatility data
- calibrate Heston parameters to a smile or surface
- inspect fit quality
- visualize outputs

## Notes For Recruiters

This repository is intended to show evidence of:

- numerical modelling in Python
- model calibration against implied-volatility data
- separation between pricing, calibration, simulation, and plotting concerns
- test-first / regression-oriented engineering practice
- ability to present results in a notebook suitable for discussion
