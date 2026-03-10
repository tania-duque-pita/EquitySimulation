# Equity Pricing

Incremental Python build-out of a Heston calibration and pricing framework for European equity options.

## Scope

The roadmap for this repository is to implement, in small commits:

- Black-Scholes pricing and implied-vol inversion
- Heston semi-analytic pricing and implied-vol smile generation
- Smile and surface calibration
- Heston Monte Carlo simulation with a robust variance scheme
- Diagnostics, plots, and regression tests

## Project Layout

The package uses a `src` layout:

```text
src/equity_pricing/
tests/
```

## Local Setup

Create a virtual environment and install the project in editable mode once Python 3.11+ is available:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
pytest
```

## Current Status

Commit 1 bootstraps packaging, dependencies, and the initial test scaffold.
