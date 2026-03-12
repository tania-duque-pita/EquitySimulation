import numpy as np
import pytest

from equity_pricing.black_scholes import (
    discount_factor,
    forward_price,
    price_bounds,
    price_european,
    vega,
)
from equity_pricing.types import FlatMarketInputs, OptionSide, VanillaOption


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


def test_call_price_bounds_match_discounted_intrinsic_and_spot_cap() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    option = VanillaOption(strike=90.0, maturity=1.0, side=OptionSide.CALL)

    lower, upper = price_bounds(option, market)

    assert lower == pytest.approx(12.409219125611283)
    assert upper == pytest.approx(98.01986733067552)


def test_put_price_bounds_match_discounted_intrinsic_and_strike_cap() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    option = VanillaOption(strike=110.0, maturity=1.0, side=OptionSide.PUT)

    lower, upper = price_bounds(option, market)

    assert lower == pytest.approx(6.615369364403023)
    assert upper == pytest.approx(104.63523669507855)


def test_price_bounds_support_vector_strikes() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    option = VanillaOption(
        strike=np.array([90.0, 100.0, 110.0]),
        maturity=1.0,
        side=OptionSide.CALL,
    )

    lower, upper = price_bounds(option, market)

    np.testing.assert_allclose(lower, np.array([10.89551496, 0.99501663, 0.0]))
    np.testing.assert_allclose(upper, np.array([100.0, 100.0, 100.0]))


def test_vega_matches_reference_value() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.05, dividend_yield=0.0)
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

    assert vega(option, market, vol=0.2) == pytest.approx(37.52403469169379)


def test_vega_is_identical_for_calls_and_puts() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)
    call = VanillaOption(strike=105.0, maturity=2.0, side=OptionSide.CALL)
    put = VanillaOption(strike=105.0, maturity=2.0, side=OptionSide.PUT)

    assert vega(call, market, vol=0.25) == pytest.approx(vega(put, market, vol=0.25))


def test_vega_rejects_non_positive_volatility() -> None:
    market = FlatMarketInputs(spot=100.0)
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

    with pytest.raises(ValueError, match="vol must be positive"):
        vega(option, market, vol=0.0)
