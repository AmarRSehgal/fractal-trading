"""Transaction-cost accounting guards.

The cost base is NOTIONAL TRADED, sum(|w_t - w_{t-1}|), which already counts
both the sells and the buys. An earlier version charged
`turnover * 2 * bps` where `turnover` had been normalized by the gross book
(= 2 for a dollar-neutral long/short), so every net-of-cost Sharpe in the
repo was flattered by a factor-of-two cost discount. These tests pin the
arithmetic against that regression.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fractal_trading.backtest import BacktestResult, cross_sectional_sort_backtest


def _flipping_panel(n_days=800, n_tickers=20, seed=0):
    """Factor whose ranking reverses every month -> near-total turnover."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    prices = pd.DataFrame(
        100 * np.exp((rng.standard_normal((n_days, n_tickers)) * 0.01).cumsum(0)),
        index=dates, columns=tickers)
    base = np.arange(n_tickers, dtype=float)
    factor = pd.DataFrame(
        [base if d.month % 2 == 0 else base[::-1] for d in dates],
        index=dates, columns=tickers)
    return factor, prices


def test_cost_is_charged_on_notional_traded():
    """Exact arithmetic: each rebalance costs sum(|dw|) * bps, no more, no less."""
    res = BacktestResult(
        portfolio_returns=pd.Series([0.0, 0.0, 0.0],
                                    index=pd.bdate_range("2020-01-31", periods=3)),
        long_returns=pd.Series(dtype=float), short_returns=pd.Series(dtype=float),
        holdings=pd.DataFrame(), turnover=1.0,
        per_rebal_turnover=[1.0, 0.5],
        per_rebal_traded_notional=[4.0, 2.0],
    )
    net = res.net_returns(10.0)
    assert net.iloc[0] == 0.0, "inception rebalance must be free"
    assert np.isclose(net.iloc[1], -4.0 * 10 / 10_000)
    assert np.isclose(net.iloc[2], -2.0 * 10 / 10_000)


def test_full_turnover_of_dollar_neutral_book_costs_four_times_bps():
    """A complete replacement of a $1-long/$1-short book trades ~4 units of
    notional, so at 10bps/side it costs ~40bps -- not the 20bps the old
    turnover-based formula charged."""
    factor, prices = _flipping_panel()
    res = cross_sectional_sort_backtest(factor, prices, n_quantiles=5,
                                        min_names_per_leg=4)
    assert np.isclose(res.turnover, 1.0, atol=0.05), (
        f"expected ~full turnover, got {res.turnover:.3f}")

    charged = (res.portfolio_returns - res.net_returns(10.0)).iloc[1:]
    traded = np.array(res.per_rebal_traded_notional)
    assert np.allclose(charged.values, traded * 10 / 10_000)
    # the headline regression: ~40bps, and specifically NOT ~20bps
    assert 35e-4 < charged.mean() < 41e-4, (
        f"full-turnover rebalance charged {charged.mean() * 1e4:.1f}bps; "
        f"the 2x-undercharge bug reports ~20bps")


def test_turnover_ratio_is_still_reported_as_fraction_of_gross():
    """The reported `turnover` metric keeps its old meaning (notional / 2 /
    gross) so historical stats CSVs stay comparable; only the cost base moved."""
    factor, prices = _flipping_panel()
    res = cross_sectional_sort_backtest(factor, prices, n_quantiles=5,
                                        min_names_per_leg=4)
    gross = res.holdings.abs().sum(axis=1).mean()
    assert np.isclose(gross, 2.0)
    ratio = np.array(res.per_rebal_traded_notional) / 2 / gross
    assert np.allclose(ratio, res.per_rebal_turnover)


def test_legacy_result_without_notional_still_charges_correctly():
    """Back-compat path: a result carrying only per_rebal_turnover must
    reconstruct the notional (ratio * 2 * gross), not re-introduce the bug."""
    res = BacktestResult(
        portfolio_returns=pd.Series([0.0, 0.0],
                                    index=pd.bdate_range("2020-01-31", periods=2)),
        long_returns=pd.Series(dtype=float), short_returns=pd.Series(dtype=float),
        holdings=pd.DataFrame(), turnover=1.0, per_rebal_turnover=[1.0],
    )
    assert np.isclose(res.net_returns(10.0).iloc[1], -4.0 * 10 / 10_000)


def test_zero_cost_is_a_no_op():
    factor, prices = _flipping_panel()
    res = cross_sectional_sort_backtest(factor, prices, n_quantiles=5,
                                        min_names_per_leg=4)
    pd.testing.assert_series_equal(res.net_returns(0.0), res.portfolio_returns)
