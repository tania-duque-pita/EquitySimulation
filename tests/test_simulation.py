import numpy as np
import pytest

from equity_pricing.simulation import (
    draw_correlated_normals,
    make_rng,
    make_time_grid,
    price_vanilla_mc,
    qe_variance_step,
    simulate_heston_paths,
)
from equity_pricing.heston import price_european as price_european_heston
from equity_pricing.types import (
    FlatMarketInputs,
    HestonParams,
    MonteCarloResult,
    OptionSide,
    VanillaOption,
)


def test_make_time_grid_includes_endpoints() -> None:
    grid = make_time_grid(1.0, 4)

    np.testing.assert_allclose(grid, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_make_time_grid_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="maturity must be positive"):
        make_time_grid(0.0, 4)

    with pytest.raises(ValueError, match="steps must be positive"):
        make_time_grid(1.0, 0)


def test_make_rng_is_deterministic_for_same_seed() -> None:
    rng1 = make_rng(123)
    rng2 = make_rng(123)

    sample1 = rng1.standard_normal(5)
    sample2 = rng2.standard_normal(5)

    np.testing.assert_allclose(sample1, sample2)


def test_draw_correlated_normals_shape_and_seed_reproducibility() -> None:
    rng1 = make_rng(42)
    rng2 = make_rng(42)

    z11, z12 = draw_correlated_normals(rng1, rho=-0.4, steps=3, n_paths=4)
    z21, z22 = draw_correlated_normals(rng2, rho=-0.4, steps=3, n_paths=4)

    assert z11.shape == (3, 4)
    assert z12.shape == (3, 4)
    np.testing.assert_allclose(z11, z21)
    np.testing.assert_allclose(z12, z22)


def test_draw_correlated_normals_empirical_correlation_is_close() -> None:
    rng = make_rng(7)
    z1, z2 = draw_correlated_normals(rng, rho=0.35, steps=400, n_paths=400)
    empirical_rho = np.corrcoef(z1.ravel(), z2.ravel())[0, 1]

    assert empirical_rho == pytest.approx(0.35, abs=0.02)


def test_draw_correlated_normals_reject_invalid_inputs() -> None:
    rng = make_rng(1)

    with pytest.raises(ValueError, match="rho must lie within"):
        draw_correlated_normals(rng, rho=1.1, steps=2, n_paths=3)

    with pytest.raises(ValueError, match="steps must be positive"):
        draw_correlated_normals(rng, rho=0.0, steps=0, n_paths=3)

    with pytest.raises(ValueError, match="n_paths must be positive"):
        draw_correlated_normals(rng, rho=0.0, steps=2, n_paths=0)


def test_qe_variance_step_returns_non_negative_values() -> None:
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.6, rho=-0.7, v0=0.05)
    variance = np.array([0.01, 0.04, 0.09])
    normal_shocks = np.array([-1.0, 0.0, 1.0])
    uniform_shocks = np.array([0.2, 0.5, 0.8])

    next_variance = qe_variance_step(
        variance,
        dt=1.0 / 12.0,
        params=params,
        normal_shocks=normal_shocks,
        uniform_shocks=uniform_shocks,
    )

    assert next_variance.shape == variance.shape
    assert np.all(next_variance >= 0.0)


def test_qe_variance_step_matches_conditional_moments_in_quadratic_branch() -> None:
    params = HestonParams(kappa=3.0, theta=0.04, sigma=0.3, rho=-0.4, v0=0.05)
    variance = np.full(200_000, 0.04)
    dt = 1.0 / 52.0
    rng = make_rng(123)

    normal_shocks = rng.standard_normal(variance.shape)
    uniform_shocks = rng.uniform(size=variance.shape)
    next_variance = qe_variance_step(
        variance,
        dt=dt,
        params=params,
        normal_shocks=normal_shocks,
        uniform_shocks=uniform_shocks,
    )

    exp_kdt = np.exp(-params.kappa * dt)
    target_mean = params.theta + (variance[0] - params.theta) * exp_kdt
    target_var = (
        variance[0]
        * params.sigma**2
        * exp_kdt
        * (1.0 - exp_kdt)
        / params.kappa
        + params.theta
        * params.sigma**2
        * (1.0 - exp_kdt) ** 2
        / (2.0 * params.kappa)
    )

    assert np.mean(next_variance) == pytest.approx(target_mean, rel=0.01)
    assert np.var(next_variance) == pytest.approx(target_var, rel=0.05)


def test_qe_variance_step_matches_conditional_moments_in_exponential_branch() -> None:
    params = HestonParams(kappa=0.8, theta=0.04, sigma=1.8, rho=-0.4, v0=0.05)
    variance = np.full(250_000, 0.04)
    dt = 2.0
    rng = make_rng(456)

    normal_shocks = rng.standard_normal(variance.shape)
    uniform_shocks = rng.uniform(size=variance.shape)
    next_variance = qe_variance_step(
        variance,
        dt=dt,
        params=params,
        normal_shocks=normal_shocks,
        uniform_shocks=uniform_shocks,
    )

    exp_kdt = np.exp(-params.kappa * dt)
    target_mean = params.theta + (variance[0] - params.theta) * exp_kdt
    target_var = (
        variance[0]
        * params.sigma**2
        * exp_kdt
        * (1.0 - exp_kdt)
        / params.kappa
        + params.theta
        * params.sigma**2
        * (1.0 - exp_kdt) ** 2
        / (2.0 * params.kappa)
    )

    assert np.mean(next_variance) == pytest.approx(target_mean, rel=0.02)
    assert np.var(next_variance) == pytest.approx(target_var, rel=0.08)
    assert np.any(next_variance == 0.0)


def test_qe_variance_step_rejects_invalid_inputs() -> None:
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.6, rho=-0.7, v0=0.05)

    with pytest.raises(ValueError, match="dt must be positive"):
        qe_variance_step(
            variance=np.array([0.04]),
            dt=0.0,
            params=params,
            normal_shocks=np.array([0.0]),
            uniform_shocks=np.array([0.5]),
        )

    with pytest.raises(ValueError, match="psi_threshold must be greater than 1.0"):
        qe_variance_step(
            variance=np.array([0.04]),
            dt=0.1,
            params=params,
            normal_shocks=np.array([0.0]),
            uniform_shocks=np.array([0.5]),
            psi_threshold=1.0,
        )

    with pytest.raises(ValueError, match="variance must be non-negative"):
        qe_variance_step(
            variance=np.array([-0.01]),
            dt=0.1,
            params=params,
            normal_shocks=np.array([0.0]),
            uniform_shocks=np.array([0.5]),
        )

    with pytest.raises(ValueError, match="must share the same shape"):
        qe_variance_step(
            variance=np.array([0.04, 0.05]),
            dt=0.1,
            params=params,
            normal_shocks=np.array([0.0]),
            uniform_shocks=np.array([0.5]),
        )

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        qe_variance_step(
            variance=np.array([0.04]),
            dt=0.1,
            params=params,
            normal_shocks=np.array([0.0]),
            uniform_shocks=np.array([1.0]),
        )


def test_simulate_heston_paths_returns_expected_shapes() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    params = HestonParams(kappa=1.8, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05)

    time_grid, spot_paths, variance_paths = simulate_heston_paths(
        market=market,
        params=params,
        maturity=1.0,
        steps=8,
        n_paths=7,
        seed=123,
    )

    assert time_grid.shape == (9,)
    assert spot_paths.shape == (9, 7)
    assert variance_paths.shape == (9, 7)
    np.testing.assert_allclose(spot_paths[0], market.spot)
    np.testing.assert_allclose(variance_paths[0], params.v0)
    assert np.all(variance_paths >= 0.0)


def test_simulate_heston_paths_is_reproducible_for_same_seed() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    params = HestonParams(kappa=1.8, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05)

    result1 = simulate_heston_paths(
        market=market,
        params=params,
        maturity=1.0,
        steps=6,
        n_paths=5,
        seed=99,
    )
    result2 = simulate_heston_paths(
        market=market,
        params=params,
        maturity=1.0,
        steps=6,
        n_paths=5,
        seed=99,
    )

    for array1, array2 in zip(result1, result2, strict=True):
        np.testing.assert_allclose(array1, array2)


def test_simulate_heston_paths_antithetic_and_plain_modes_both_work() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.4, rho=-0.5, v0=0.04)

    _, spot_plain, variance_plain = simulate_heston_paths(
        market=market,
        params=params,
        maturity=0.5,
        steps=4,
        n_paths=5,
        seed=321,
        antithetic=False,
    )
    _, spot_anti, variance_anti = simulate_heston_paths(
        market=market,
        params=params,
        maturity=0.5,
        steps=4,
        n_paths=5,
        seed=321,
        antithetic=True,
    )

    assert spot_plain.shape == (5, 5)
    assert variance_plain.shape == (5, 5)
    assert spot_anti.shape == (5, 5)
    assert variance_anti.shape == (5, 5)
    assert np.all(variance_plain >= 0.0)
    assert np.all(variance_anti >= 0.0)


def test_simulate_heston_paths_mean_terminal_spot_is_close_to_forward_when_vol_of_vol_is_small() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)
    params = HestonParams(kappa=6.0, theta=0.04, sigma=0.05, rho=-0.3, v0=0.04)

    _, spot_paths, variance_paths = simulate_heston_paths(
        market=market,
        params=params,
        maturity=1.0,
        steps=64,
        n_paths=20_000,
        seed=2024,
    )

    terminal_mean = float(np.mean(spot_paths[-1]))
    target_forward = market.spot * np.exp(
        (market.risk_free_rate - market.dividend_yield) * 1.0
    )

    assert terminal_mean == pytest.approx(target_forward, rel=0.02)
    assert np.all(variance_paths >= 0.0)


def test_simulate_heston_paths_rejects_invalid_inputs() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    params = HestonParams(kappa=1.8, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05)

    with pytest.raises(ValueError, match="maturity must be positive"):
        simulate_heston_paths(
            market=market,
            params=params,
            maturity=0.0,
            steps=8,
            n_paths=7,
        )

    with pytest.raises(ValueError, match="steps must be positive"):
        simulate_heston_paths(
            market=market,
            params=params,
            maturity=1.0,
            steps=0,
            n_paths=7,
        )

    with pytest.raises(ValueError, match="n_paths must be positive"):
        simulate_heston_paths(
            market=market,
            params=params,
            maturity=1.0,
            steps=8,
            n_paths=0,
        )


def test_price_vanilla_mc_returns_structured_result() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    params = HestonParams(kappa=1.8, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05)
    option = VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL)

    result = price_vanilla_mc(
        option=option,
        market=market,
        params=params,
        steps=32,
        n_paths=2_000,
        seed=123,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.n_paths == 2_000
    assert result.discounted_payoffs.shape == (2_000,)
    assert result.standard_error > 0.0
    assert result.confidence_interval[0] <= result.price <= result.confidence_interval[1]


def test_price_vanilla_mc_matches_discounted_intrinsic_in_nearly_deterministic_case() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01)
    params = HestonParams(kappa=20.0, theta=1.0e-6, sigma=1.0e-6, rho=0.0, v0=1.0e-6)
    option = VanillaOption(strike=95.0, maturity=1.0, side=OptionSide.CALL)

    result = price_vanilla_mc(
        option=option,
        market=market,
        params=params,
        steps=16,
        n_paths=4_000,
        seed=999,
    )

    deterministic_terminal_spot = market.spot * np.exp(
        (market.risk_free_rate - market.dividend_yield) * option.maturity
    )
    target_price = np.exp(-market.risk_free_rate * option.maturity) * max(
        deterministic_terminal_spot - float(option.strike),
        0.0,
    )

    assert result.price == pytest.approx(target_price, rel=0.02)
    assert result.standard_error < 0.1


def test_price_vanilla_mc_put_payoff_is_supported() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.6, rho=-0.7, v0=0.05)
    option = VanillaOption(strike=105.0, maturity=0.75, side=OptionSide.PUT)

    result = price_vanilla_mc(
        option=option,
        market=market,
        params=params,
        steps=24,
        n_paths=3_000,
        seed=77,
    )

    assert result.price >= 0.0
    assert np.all(result.discounted_payoffs >= 0.0)


def test_price_vanilla_mc_rejects_invalid_inputs() -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    params = HestonParams(kappa=1.8, theta=0.04, sigma=0.5, rho=-0.6, v0=0.05)

    with pytest.raises(ValueError, match="requires a scalar strike"):
        price_vanilla_mc(
            option=VanillaOption(
                strike=np.array([95.0, 100.0]),
                maturity=1.0,
                side=OptionSide.CALL,
            ),
            market=market,
            params=params,
            steps=16,
            n_paths=2_000,
        )

    with pytest.raises(ValueError, match="confidence_level must lie strictly between 0 and 1"):
        price_vanilla_mc(
            option=VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL),
            market=market,
            params=params,
            steps=16,
            n_paths=2_000,
            confidence_level=1.0,
        )


@pytest.mark.parametrize(
    ("option", "seed"),
    [
        (VanillaOption(strike=100.0, maturity=1.0, side=OptionSide.CALL), 2025),
        (VanillaOption(strike=100.0, maturity=0.5, side=OptionSide.CALL), 2026),
    ],
)
def test_price_vanilla_mc_confidence_interval_contains_heston_price(
    option: VanillaOption,
    seed: int,
) -> None:
    market = FlatMarketInputs(spot=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    params = HestonParams(kappa=2.5, theta=0.04, sigma=0.2, rho=-0.3, v0=0.04)

    mc_result = price_vanilla_mc(
        option=option,
        market=market,
        params=params,
        steps=48,
        n_paths=12_000,
        seed=seed,
        confidence_level=0.99,
    )
    analytic_price = price_european_heston(
        option,
        market,
        params,
        upper_limit=150.0,
        abs_tol=1.0e-7,
        rel_tol=1.0e-7,
    )

    assert mc_result.confidence_interval[0] <= analytic_price <= mc_result.confidence_interval[1]
    assert abs(mc_result.price - analytic_price) <= 3.0 * mc_result.standard_error
