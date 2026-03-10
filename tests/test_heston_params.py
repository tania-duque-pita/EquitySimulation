import numpy as np
import pytest

from equity_pricing import HestonParams


def test_heston_params_accept_valid_inputs() -> None:
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.6, rho=-0.7, v0=0.05)

    np.testing.assert_allclose(
        params.as_array(),
        np.array([1.5, 0.04, 0.6, -0.7, 0.05]),
    )


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("kappa", {"kappa": 0.0, "theta": 0.04, "sigma": 0.6, "rho": -0.7, "v0": 0.05}),
        ("theta", {"kappa": 1.5, "theta": 0.0, "sigma": 0.6, "rho": -0.7, "v0": 0.05}),
        ("sigma", {"kappa": 1.5, "theta": 0.04, "sigma": 0.0, "rho": -0.7, "v0": 0.05}),
        ("rho", {"kappa": 1.5, "theta": 0.04, "sigma": 0.6, "rho": 1.5, "v0": 0.05}),
        ("v0", {"kappa": 1.5, "theta": 0.04, "sigma": 0.6, "rho": -0.7, "v0": 5.0}),
    ],
)
def test_heston_params_reject_out_of_bounds_values(
    field_name: str,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=field_name):
        HestonParams(**kwargs)


def test_heston_params_to_unconstrained_and_back_roundtrip() -> None:
    params = HestonParams(kappa=2.25, theta=0.09, sigma=1.1, rho=-0.35, v0=0.08)

    roundtripped = HestonParams.from_unconstrained(params.to_unconstrained())

    np.testing.assert_allclose(roundtripped.as_array(), params.as_array(), rtol=1e-10, atol=1e-10)


def test_heston_params_from_unconstrained_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        HestonParams.from_unconstrained(np.array([0.0, 1.0]))


def test_heston_params_from_unconstrained_respects_bounds() -> None:
    params = HestonParams.from_unconstrained(np.array([-50.0, 50.0, 0.0, -50.0, 50.0]))

    assert HestonParams.BOUNDS["kappa"][0] <= params.kappa <= HestonParams.BOUNDS["kappa"][1]
    assert HestonParams.BOUNDS["theta"][0] <= params.theta <= HestonParams.BOUNDS["theta"][1]
    assert HestonParams.BOUNDS["sigma"][0] <= params.sigma <= HestonParams.BOUNDS["sigma"][1]
    assert HestonParams.BOUNDS["rho"][0] <= params.rho <= HestonParams.BOUNDS["rho"][1]
    assert HestonParams.BOUNDS["v0"][0] <= params.v0 <= HestonParams.BOUNDS["v0"][1]
