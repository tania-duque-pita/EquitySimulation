import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from equity_pricing.examples import run_end_to_end_example


def test_run_end_to_end_example_smoke() -> None:
    results = run_end_to_end_example()

    assert results["surface_result"].success
    assert results["smile_result"].success
    assert results["surface_result"].rmse < 1.0e-3
    assert results["mc_result"].price > 0.0
    assert set(results["figures"]) == {
        "smile_fit",
        "surface_fit",
        "residual_heatmap",
    }

    for figure in results["figures"].values():
        plt.close(figure)
