import numpy as np
import pytest

from equity_pricing import MarketSmile, MarketSurface, SmileQuote


def _make_smile(maturity: float, vols: tuple[float, float]) -> MarketSmile:
    return MarketSmile(
        quotes=(
            SmileQuote(strike=95.0, maturity=maturity, implied_vol=vols[0]),
            SmileQuote(strike=105.0, maturity=maturity, implied_vol=vols[1]),
        )
    )


def test_market_surface_sorts_smiles_by_maturity() -> None:
    surface = MarketSurface(
        smiles=(
            _make_smile(2.0, (0.23, 0.21)),
            _make_smile(0.5, (0.27, 0.24)),
            _make_smile(1.0, (0.25, 0.22)),
        )
    )

    assert [smile.maturity for smile in surface.smiles] == [0.5, 1.0, 2.0]
    np.testing.assert_allclose(surface.maturities, np.array([0.5, 1.0, 2.0]))


def test_market_surface_rejects_empty_smiles() -> None:
    with pytest.raises(ValueError, match="smiles must not be empty"):
        MarketSurface(smiles=())


def test_market_surface_rejects_duplicate_maturities() -> None:
    with pytest.raises(ValueError, match="unique"):
        MarketSurface(
            smiles=(
                _make_smile(1.0, (0.24, 0.22)),
                _make_smile(1.0, (0.25, 0.23)),
            )
        )
