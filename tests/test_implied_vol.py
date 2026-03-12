import numpy as np
import pytest

from equity_pricing.black_scholes import price_european
from equity_pricing.implied_vol import implied_vol_from_price
from equity_pricing.types import FlatMarketInputs, OptionSide, VanillaOption


def test_implied_vol_recovers_call_volatility() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)
    option = VanillaOption(strike=105.0, maturity=1.25, side=OptionSide.CALL)
    target_vol = 0.24
    price = price_european(option, market, target_vol)

    implied_vol = implied_vol_from_price(price, option, market)

    assert implied_vol == pytest.approx(target_vol)


def test_implied_vol_recovers_put_volatility() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.00)
    option = VanillaOption(strike=95.0, maturity=0.75, side=OptionSide.PUT)
    target_vol = 0.31
    price = price_european(option, market, target_vol)

    implied_vol = implied_vol_from_price(price, option, market)

    assert implied_vol == pytest.approx(target_vol)


def test_implied_vol_rejects_price_below_no_arbitrage_bounds() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.05, dividend_yield=0.0)
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_vol_from_price(0.5, option, market)


def test_implied_vol_rejects_price_above_no_arbitrage_bounds() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_vol_from_price(150.0, option, market)


def test_implied_vol_uses_custom_initial_upper_bracket() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    option = VanillaOption(strike=150.0, maturity=2.0, side=OptionSide.CALL)
    target_vol = 1.8
    price = price_european(option, market, target_vol)

    implied_vol = implied_vol_from_price(price, option, market, upper_vol=0.5)

    assert implied_vol == pytest.approx(target_vol)


def test_implied_vol_rejects_vector_strikes() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    option = VanillaOption(
        strike=np.array([90.0, 100.0]),
        maturity=1.0,
        side=OptionSide.CALL,
    )

    with pytest.raises(TypeError, match="scalar strikes"):
        implied_vol_from_price(10.0, option, market)
