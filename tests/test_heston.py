import math

import numpy as np
import pytest

from equity_pricing.black_scholes import price_european
from equity_pricing.heston import (
    heston_characteristic_function,
    heston_lewis_integrand,
    integrate_heston_integrand,
    model_smile,
    model_surface,
    price_european as price_european_heston,
)
from equity_pricing.types import (
    FlatMarketInputs,
    HestonParams,
    MarketSurface,
    OptionSide,
    VanillaOption,
)


@pytest.fixture
def sample_market() -> FlatMarketInputs:
    return FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)


@pytest.fixture
def sample_params() -> HestonParams:
    return HestonParams(kappa=1.7, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05)


def test_heston_characteristic_function_is_one_at_zero(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    value = heston_characteristic_function(0.0, 1.25, sample_market, sample_params)

    assert value == pytest.approx(1.0 + 0.0j)


def test_heston_characteristic_function_matches_first_moment(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    maturity = 1.25
    value = heston_characteristic_function(-1j, maturity, sample_market, sample_params)
    expected = sample_market.spot * np.exp(
        (sample_market.risk_free_rate - sample_market.dividend_yield) * maturity
    )

    assert value == pytest.approx(expected + 0.0j)


def test_heston_characteristic_function_supports_vector_inputs(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    values = heston_characteristic_function(
        np.array([0.1, 0.5, 1.0], dtype=np.complex128),
        1.25,
        sample_market,
        sample_params,
    )

    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)


def test_heston_characteristic_function_matches_regression_values(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    values = heston_characteristic_function(
        np.array([0.1, 0.5, 1.0], dtype=np.complex128),
        1.25,
        sample_market,
        sample_params,
    )

    expected = np.array(
        [
            0.895660886048876 + 0.444046122723760j,
            -0.662498574510127 + 0.738866177036545j,
            -0.102854788750759 - 0.964895892195100j,
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)


def test_heston_characteristic_function_rejects_non_positive_maturity(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    with pytest.raises(ValueError, match="maturity must be positive"):
        heston_characteristic_function(0.1, 0.0, sample_market, sample_params)


def test_heston_lewis_integrand_returns_finite_values(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    values = heston_lewis_integrand(
        np.array([0.0, 0.25, 1.0, 5.0]),
        log_moneyness=0.0,
        maturity=1.25,
        market=sample_market,
        params=sample_params,
    )

    assert np.all(np.isfinite(values))


def test_heston_lewis_integrand_rejects_negative_u(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        heston_lewis_integrand(
            -0.1,
            log_moneyness=0.0,
            maturity=1.25,
            market=sample_market,
            params=sample_params,
        )


def test_integrate_heston_integrand_matches_known_integral() -> None:
    value, error = integrate_heston_integrand(lambda u: math.exp(-u), upper_limit=50.0)

    assert value == pytest.approx(1.0, rel=1e-8, abs=1e-8)
    assert error < 1e-8


def test_integrate_heston_integrand_rejects_non_positive_upper_limit() -> None:
    with pytest.raises(ValueError, match="upper_limit must be positive"):
        integrate_heston_integrand(lambda u: u, upper_limit=0.0)


def test_heston_lewis_integrand_can_be_integrated(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    value, error = integrate_heston_integrand(
        lambda u: heston_lewis_integrand(
            u,
            log_moneyness=0.0,
            maturity=1.25,
            market=sample_market,
            params=sample_params,
        ),
        upper_limit=100.0,
    )

    assert math.isfinite(value)
    assert value > 0.0
    assert error >= 0.0


def test_heston_call_price_matches_black_scholes_in_near_deterministic_limit(
    sample_market: FlatMarketInputs,
) -> None:
    option = VanillaOption(strike=100.0, maturity=1.25, side=OptionSide.CALL)
    variance = 0.04
    params = HestonParams(kappa=20.0, theta=variance, sigma=1.0e-3, rho=0.0, v0=variance)

    heston_price = price_european_heston(option, sample_market, params, upper_limit=120.0)
    black_scholes_price = price_european(option, sample_market, vol=math.sqrt(variance))

    assert heston_price == pytest.approx(black_scholes_price, abs=5.0e-3)


def test_heston_call_price_matches_regression_value(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    option = VanillaOption(strike=100.0, maturity=1.25, side=OptionSide.CALL)

    price = price_european_heston(option, sample_market, sample_params)

    assert price == pytest.approx(9.680777762358808, rel=1e-10, abs=1e-10)


def test_heston_put_price_matches_put_call_parity(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    strike = 100.0
    maturity = 1.25
    call = VanillaOption(strike=strike, maturity=maturity, side=OptionSide.CALL)
    put = VanillaOption(strike=strike, maturity=maturity, side=OptionSide.PUT)

    call_price = price_european_heston(call, sample_market, sample_params)
    put_price = price_european_heston(put, sample_market, sample_params)
    parity_rhs = (
        strike * math.exp(-sample_market.risk_free_rate * maturity)
        - sample_market.spot * math.exp(-sample_market.dividend_yield * maturity)
    )

    assert put_price - call_price == pytest.approx(parity_rhs, abs=1e-10)


def test_heston_price_supports_vector_strikes(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    option = VanillaOption(
        strike=np.array([95.0, 100.0]),
        maturity=1.25,
        side=OptionSide.CALL,
    )

    prices = price_european_heston(option, sample_market, sample_params)

    assert isinstance(prices, np.ndarray)
    assert prices.shape == (2,)
    np.testing.assert_allclose(prices, np.array([12.71343440, 9.68077776]), rtol=1e-8, atol=1e-8)


def test_heston_put_price_supports_vector_strikes(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    option = VanillaOption(
        strike=np.array([95.0, 100.0]),
        maturity=1.25,
        side=OptionSide.PUT,
    )

    prices = price_european_heston(option, sample_market, sample_params)

    assert isinstance(prices, np.ndarray)
    assert prices.shape == (2,)
    np.testing.assert_allclose(prices, np.array([5.45912403, 7.24243949]), rtol=1e-8, atol=1e-8)


def test_heston_model_smile_returns_implied_vols_for_strike_grid(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    strikes = np.array([95.0, 100.0, 105.0])

    smile = model_smile(strikes, 1.25, sample_market, sample_params)

    assert smile.shape == (3,)
    assert np.all(np.isfinite(smile))
    np.testing.assert_allclose(smile, np.array([0.2033128, 0.19359116, 0.18470877]), rtol=1e-8, atol=1e-8)


def test_heston_model_smile_put_and_call_match_implied_vols(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    strikes = np.array([95.0, 100.0])

    call_smile = model_smile(strikes, 1.25, sample_market, sample_params, side=OptionSide.CALL)
    put_smile = model_smile(strikes, 1.25, sample_market, sample_params, side=OptionSide.PUT)

    np.testing.assert_allclose(call_smile, put_smile, rtol=1e-10, atol=1e-10)


def test_heston_model_smile_returns_nan_on_inversion_failure(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_inversion(*args, **kwargs) -> float:
        raise RuntimeError("boom")

    monkeypatch.setattr("equity_pricing.heston.implied_vol_from_price", failing_inversion)

    smile = model_smile(np.array([100.0]), 1.25, sample_market, sample_params)

    assert np.isnan(smile[0])


def test_heston_model_smile_can_raise_on_inversion_failure(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_inversion(*args, **kwargs) -> float:
        raise RuntimeError("boom")

    monkeypatch.setattr("equity_pricing.heston.implied_vol_from_price", failing_inversion)

    with pytest.raises(RuntimeError, match="boom"):
        model_smile(np.array([100.0]), 1.25, sample_market, sample_params, fill_value=0.0)


def test_heston_model_smile_rejects_non_1d_strikes(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    with pytest.raises(ValueError, match="1D"):
        model_smile(np.array([[100.0, 105.0]]), 1.25, sample_market, sample_params)


def test_heston_model_surface_returns_ordered_surface(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    surface = model_surface(
        strikes_by_expiry=(
            np.array([95.0, 100.0]),
            np.array([90.0, 100.0, 110.0]),
        ),
        maturities=np.array([1.5, 0.75]),
        market=sample_market,
        params=sample_params,
    )

    assert isinstance(surface, MarketSurface)
    np.testing.assert_allclose(surface.maturities, np.array([0.75, 1.5]))
    assert surface.smiles[0].strikes.shape == (3,)
    assert surface.smiles[1].strikes.shape == (2,)


def test_heston_model_surface_matches_per_expiry_smiles(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    strikes_by_expiry = (
        np.array([95.0, 100.0]),
        np.array([95.0, 100.0]),
    )
    maturities = np.array([0.75, 1.5])

    surface = model_surface(strikes_by_expiry, maturities, sample_market, sample_params)

    for smile, strikes, maturity in zip(surface.smiles, strikes_by_expiry, maturities, strict=True):
        expected = model_smile(strikes, maturity, sample_market, sample_params)
        np.testing.assert_allclose(smile.implied_vols, expected, rtol=1e-10, atol=1e-10)


def test_heston_model_surface_rejects_length_mismatch(
    sample_market: FlatMarketInputs,
    sample_params: HestonParams,
) -> None:
    with pytest.raises(ValueError, match="same length"):
        model_surface(
            strikes_by_expiry=(np.array([95.0, 100.0]),),
            maturities=np.array([0.5, 1.0]),
            market=sample_market,
            params=sample_params,
        )
