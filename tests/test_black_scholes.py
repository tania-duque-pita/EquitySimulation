import numpy as np
import pytest

from equity_pricing import (
    FlatMarketInputs,
    OptionSide,
    VanillaOption,
    discount_factor,
    forward_price,
    price_european,
)


def test_discount_factor_matches_continuous_discounting() -> None:
    assert discount_factor(0.05, 2.0) == pytest.approx(0.9048374180359595)


def test_forward_price_matches_cost_of_carry_formula() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    assert forward_price(market, 1.5) == pytest.approx(104.6027859908717)


def test_call_price_matches_reference_value() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.05, dividend_yield=0.0)
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

    assert price_european(option, market, vol=0.2) == pytest.approx(10.450583572185565)


def test_put_price_matches_reference_value() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.05, dividend_yield=0.0)
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.PUT)

    assert price_european(option, market, vol=0.2) == pytest.approx(5.573526022256971)


def test_price_european_supports_vector_strikes() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    option = VanillaOption(
        strike=np.array([90.0, 100.0, 110.0]),
        maturity=1.0,
        side=OptionSide.CALL,
    )

    prices = price_european(option, market, vol=0.2)

    assert isinstance(prices, np.ndarray)
    assert prices.shape == (3,)
    np.testing.assert_allclose(prices, np.array([14.19292021, 8.43331869, 4.61011457]))


def test_price_european_rejects_non_positive_volatility() -> None:
    market = FlatMarketInputs(spot=100.0)
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

    with pytest.raises(ValueError, match="vol must be positive"):
        price_european(option, market, vol=0.0)
