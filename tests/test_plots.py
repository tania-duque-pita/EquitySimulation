import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from equity_pricing import MarketSmile, SmileQuote, plot_market_smile


@pytest.fixture
def sample_smile() -> MarketSmile:
    return MarketSmile(
        quotes=(
            SmileQuote(strike=90.0, maturity=1.0, implied_vol=0.27),
            SmileQuote(strike=100.0, maturity=1.0, implied_vol=0.24),
            SmileQuote(strike=110.0, maturity=1.0, implied_vol=0.22),
        )
    )


def test_plot_market_smile_returns_figure_and_axes(sample_smile: MarketSmile) -> None:
    figure, axes = plot_market_smile(sample_smile)

    assert figure.axes == [axes]
    assert axes.get_xlabel() == "Strike"
    assert axes.get_ylabel() == "Implied Volatility"
    assert axes.get_title() == "Market Smile (T=1.00)"

    plt.close(figure)


def test_plot_market_smile_uses_custom_title(sample_smile: MarketSmile) -> None:
    figure, axes = plot_market_smile(sample_smile, title="Test Smile")

    assert axes.get_title() == "Test Smile"

    plt.close(figure)


def test_plot_market_smile_draws_single_line(sample_smile: MarketSmile) -> None:
    figure, axes = plot_market_smile(sample_smile)
    line = axes.lines[0]

    assert len(axes.lines) == 1
    assert list(line.get_xdata()) == [90.0, 100.0, 110.0]
    assert list(line.get_ydata()) == [0.27, 0.24, 0.22]

    plt.close(figure)
