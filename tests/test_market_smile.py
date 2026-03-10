import numpy as np
import pytest

from equity_pricing import MarketSmile, SmileQuote


def test_smile_quote_accepts_valid_inputs() -> None:
    quote = SmileQuote(strike=100.0, maturity=1.0, implied_vol=0.2)

    assert quote.strike == 100.0
    assert quote.maturity == 1.0
    assert quote.implied_vol == 0.2


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("strike", {"strike": 0.0, "maturity": 1.0, "implied_vol": 0.2}),
        ("maturity", {"strike": 100.0, "maturity": 0.0, "implied_vol": 0.2}),
        ("implied_vol", {"strike": 100.0, "maturity": 1.0, "implied_vol": 0.0}),
    ],
)
def test_smile_quote_rejects_non_positive_values(
    field_name: str,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=field_name):
        SmileQuote(**kwargs)


def test_market_smile_sorts_quotes_by_strike() -> None:
    smile = MarketSmile(
        quotes=(
            SmileQuote(strike=110.0, maturity=1.0, implied_vol=0.24),
            SmileQuote(strike=90.0, maturity=1.0, implied_vol=0.27),
            SmileQuote(strike=100.0, maturity=1.0, implied_vol=0.25),
        )
    )

    assert [quote.strike for quote in smile.quotes] == [90.0, 100.0, 110.0]
    np.testing.assert_allclose(smile.strikes, np.array([90.0, 100.0, 110.0]))
    np.testing.assert_allclose(smile.implied_vols, np.array([0.27, 0.25, 0.24]))


def test_market_smile_exposes_shared_maturity() -> None:
    smile = MarketSmile(
        quotes=(
            SmileQuote(strike=95.0, maturity=0.5, implied_vol=0.23),
            SmileQuote(strike=105.0, maturity=0.5, implied_vol=0.21),
        )
    )

    assert smile.maturity == 0.5


def test_market_smile_rejects_empty_quotes() -> None:
    with pytest.raises(ValueError, match="quotes must not be empty"):
        MarketSmile(quotes=())


def test_market_smile_rejects_mixed_maturities() -> None:
    with pytest.raises(ValueError, match="same maturity"):
        MarketSmile(
            quotes=(
                SmileQuote(strike=95.0, maturity=0.5, implied_vol=0.23),
                SmileQuote(strike=105.0, maturity=1.0, implied_vol=0.21),
            )
        )


def test_market_smile_rejects_duplicate_strikes() -> None:
    with pytest.raises(ValueError, match="unique"):
        MarketSmile(
            quotes=(
                SmileQuote(strike=100.0, maturity=1.0, implied_vol=0.22),
                SmileQuote(strike=100.0, maturity=1.0, implied_vol=0.24),
            )
        )
