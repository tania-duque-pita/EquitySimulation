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
