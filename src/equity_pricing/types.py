"""Core domain types shared across pricing and calibration modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


def _require_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")


def _logit(value: float, lower: float, upper: float) -> float:
    scaled = (value - lower) / (upper - lower)
    return float(np.log(scaled / (1.0 - scaled)))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


class OptionSide(str, Enum):
    """Vanilla European option side."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class VanillaOption:
    """European vanilla equity option contract."""

    strike: float | np.ndarray
    maturity: float
    side: OptionSide

    def __post_init__(self) -> None:
        strikes = np.asarray(self.strike, dtype=float)
        if np.any(strikes <= 0.0):
            raise ValueError("strike must be positive.")
        _require_positive(self.maturity, "maturity")


@dataclass(frozen=True, slots=True)
class FlatMarketInputs:
    """Flat spot, rate, and dividend inputs for equity option pricing."""

    spot: float
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        _require_positive(self.spot, "spot")
        _require_non_negative(self.risk_free_rate + 1.0, "1 + risk_free_rate")
        _require_non_negative(self.dividend_yield + 1.0, "1 + dividend_yield")


@dataclass(frozen=True, slots=True)
class SmileQuote:
    """Single market implied-volatility quote for one strike and expiry."""

    strike: float
    maturity: float
    implied_vol: float

    def __post_init__(self) -> None:
        _require_positive(self.strike, "strike")
        _require_positive(self.maturity, "maturity")
        _require_positive(self.implied_vol, "implied_vol")


@dataclass(frozen=True, slots=True)
class MarketSmile:
    """Collection of market quotes for a single expiry, sorted by strike."""

    quotes: Tuple[SmileQuote, ...]

    def __post_init__(self) -> None:
        if not self.quotes:
            raise ValueError("quotes must not be empty.")

        maturities = {quote.maturity for quote in self.quotes}
        if len(maturities) != 1:
            raise ValueError("all smile quotes must share the same maturity.")

        sorted_quotes = tuple(sorted(self.quotes, key=lambda quote: quote.strike))
        object.__setattr__(self, "quotes", sorted_quotes)

        strikes = [quote.strike for quote in sorted_quotes]
        if len(set(strikes)) != len(strikes):
            raise ValueError("smile strikes must be unique.")

    @property
    def maturity(self) -> float:
        return self.quotes[0].maturity

    @property
    def strikes(self) -> np.ndarray:
        return np.array([quote.strike for quote in self.quotes], dtype=float)

    @property
    def implied_vols(self) -> np.ndarray:
        return np.array([quote.implied_vol for quote in self.quotes], dtype=float)


@dataclass(frozen=True, slots=True)
class MarketSurface:
    """Ordered collection of market smiles across expiries."""

    smiles: Tuple[MarketSmile, ...]

    def __post_init__(self) -> None:
        if not self.smiles:
            raise ValueError("smiles must not be empty.")

        sorted_smiles = tuple(sorted(self.smiles, key=lambda smile: smile.maturity))
        object.__setattr__(self, "smiles", sorted_smiles)

        maturities = [smile.maturity for smile in sorted_smiles]
        if len(set(maturities)) != len(maturities):
            raise ValueError("surface maturities must be unique.")

    @property
    def maturities(self) -> np.ndarray:
        return np.array([smile.maturity for smile in self.smiles], dtype=float)


@dataclass(frozen=True, slots=True)
class HestonParams:
    """Heston model parameters with hard bounds and optimizer transforms."""

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    # Upper bounds are finite to support stable optimizer transforms.
    BOUNDS = {
        "kappa": (1.0e-6, 25.0),
        "theta": (1.0e-6, 4.0),
        "sigma": (1.0e-6, 10.0),
        "rho": (-0.999, 0.999),
        "v0": (1.0e-6, 4.0),
    }

    def __post_init__(self) -> None:
        for name, (lower, upper) in self.BOUNDS.items():
            value = getattr(self, name)
            if not lower <= value <= upper:
                raise ValueError(
                    f"{name} must lie within [{lower}, {upper}], got {value!r}."
                )

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.kappa, self.theta, self.sigma, self.rho, self.v0],
            dtype=float,
        )

    def to_unconstrained(self) -> np.ndarray:
        return np.array(
            [
                _logit(self.kappa, *self.BOUNDS["kappa"]),
                _logit(self.theta, *self.BOUNDS["theta"]),
                _logit(self.sigma, *self.BOUNDS["sigma"]),
                _logit(self.rho, *self.BOUNDS["rho"]),
                _logit(self.v0, *self.BOUNDS["v0"]),
            ],
            dtype=float,
        )

    @classmethod
    def from_unconstrained(cls, values: np.ndarray) -> "HestonParams":
        vector = np.asarray(values, dtype=float)
        if vector.shape != (5,):
            raise ValueError(f"values must have shape (5,), got {vector.shape!r}.")

        def _bounded(component: float, name: str) -> float:
            lower, upper = cls.BOUNDS[name]
            return lower + (upper - lower) * _sigmoid(float(component))

        return cls(
            kappa=_bounded(vector[0], "kappa"),
            theta=_bounded(vector[1], "theta"),
            sigma=_bounded(vector[2], "sigma"),
            rho=_bounded(vector[3], "rho"),
            v0=_bounded(vector[4], "v0"),
        )


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    """Settings for Heston calibration objectives and optimizers."""

    nan_penalty: float = 1.0
    upper_limit: float = 200.0
    abs_tol: float = 1.0e-8
    rel_tol: float = 1.0e-8
    integration_limit: int = 200
    n_restarts: int = 4
    enable_feller_penalty: bool = False
    feller_penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        _require_positive(self.nan_penalty, "nan_penalty")
        _require_positive(self.upper_limit, "upper_limit")
        _require_positive(self.abs_tol, "abs_tol")
        _require_positive(self.rel_tol, "rel_tol")
        if self.integration_limit <= 0:
            raise ValueError(
                f"integration_limit must be positive, got {self.integration_limit!r}."
            )
        if self.n_restarts <= 0:
            raise ValueError(f"n_restarts must be positive, got {self.n_restarts!r}.")
        _require_positive(self.feller_penalty_weight, "feller_penalty_weight")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Result of a Heston calibration run."""

    params: HestonParams
    residuals: np.ndarray
    objective_value: float
    model_vols: np.ndarray
    market_vols: np.ndarray
    rmse: float
    mae: float
    max_abs_error: float
    success: bool
    nfev: int
    message: str
    n_restarts: int
