import numpy as np
import pytest

from equity_pricing import draw_correlated_normals, make_rng, make_time_grid


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
