import numpy as np
import pytest

from equity_pricing import FlatMarketInputs, HestonParams, heston_characteristic_function


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
