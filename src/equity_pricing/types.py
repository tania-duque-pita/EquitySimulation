"""Core domain types shared across pricing and calibration modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _require_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")


class OptionSide(str, Enum):
    """Vanilla European option side."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class VanillaOption:
    """European vanilla equity option contract."""

    strike: float
    maturity: float
    side: OptionSide

    def __post_init__(self) -> None:
        _require_positive(self.strike, "strike")
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
