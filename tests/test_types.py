import re

import pytest

from equity_pricing import FlatMarketInputs, OptionSide, VanillaOption


def test_vanilla_option_accepts_valid_inputs() -> None:
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

    assert option.strike == 100.0
    assert option.maturity == 1.0
    assert option.side is OptionSide.CALL


@pytest.mark.parametrize("strike", [0.0, -1.0])
def test_vanilla_option_rejects_non_positive_strike(strike: float) -> None:
    with pytest.raises(ValueError, match="strike must be positive"):
        VanillaOption(strike=strike, maturity=1.0, side=OptionSide.PUT)


@pytest.mark.parametrize("maturity", [0.0, -0.25])
def test_vanilla_option_rejects_non_positive_maturity(maturity: float) -> None:
    with pytest.raises(ValueError, match="maturity must be positive"):
        VanillaOption(strike=100.0, maturity=maturity, side=OptionSide.PUT)


def test_flat_market_inputs_accept_valid_values() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)

    assert market.spot == 100.0
    assert market.risk_free_rate == 0.03
    assert market.dividend_yield == 0.01


@pytest.mark.parametrize("spot", [0.0, -10.0])
def test_flat_market_inputs_reject_non_positive_spot(spot: float) -> None:
    with pytest.raises(ValueError, match="spot must be positive"):
        FlatMarketInputs(spot=spot)


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("1 + risk_free_rate", {"spot": 100.0, "risk_free_rate": -1.1}),
        ("1 + dividend_yield", {"spot": 100.0, "dividend_yield": -1.5}),
    ],
)
def test_flat_market_inputs_reject_extreme_negative_carry_values(
    field_name: str,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=re.escape(field_name)):
        FlatMarketInputs(**kwargs)
