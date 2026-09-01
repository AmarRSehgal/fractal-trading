"""Leakage guards for the walk-forward embargo and the bootstrap CIs.

The single most expensive bug this project has hit was a 22-day embargo
omission that turned a true Sharpe of 0.015 into an apparent 1.84. These
tests pin the invariant so it cannot silently regress.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib

import numpy as np
import pandas as pd

from fractal_trading.backtest import (
    cross_sectional_sort_backtest,
    paired_sharpe_diff_ci,
    sharpe_ci,
    stationary_bootstrap_indices,
)

wf = importlib.import_module("07_gbm_walkforward")

TARGET_HORIZON = 21   # make_panel target is log_p.shift(-21) - log_p


def _panel(n_days=1600, n_tickers=12, seed=0):
    """Synthetic panel with the same columns walk_forward_predict expects."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for t in range(n_tickers):
        for d in dates:
            rows.append({
                "date": d, "ticker": f"T{t}",
                "mom_12_1": rng.standard_normal(), "ret_1m": rng.standard_normal(),
                "vol_60": abs(rng.standard_normal()), "vol_252": abs(rng.standard_normal()),
                "fd_z": rng.standard_normal(), "target": rng.standard_normal() * 0.05,
            })
    return pd.DataFrame(rows)


def test_embargo_excludes_unobservable_training_targets(monkeypatch):
    """No training row may have a target window reaching past the test date.

    A training row at d_train has a target spanning [d_train, d_train + 21
    trading days]. At test date d_test that target is only observable if at
    least 21 business days separate them. We install a spy model that records
    the real training set walk_forward_predict hands it, then assert the gap.
    """
    panel = _panel()
    seen = []

    class _Spy:
        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(wf, "USE_LGB", True)
    monkeypatch.setattr(
        wf, "lgb", type("m", (), {"LGBMRegressor": _Spy}), raising=False
    )

    preds = wf.walk_forward_predict(
        panel, ["mom_12_1", "ret_1m", "vol_60", "vol_252", "fd_z"],
        train_years=5, embargo_days=22)
    assert not preds.empty, "walk-forward produced no predictions"

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    for date in pd.DatetimeIndex(sorted(preds["date"].unique())):
        embargo_end = date - pd.offsets.BDay(22)
        train = panel[(panel["date"] >= date - pd.DateOffset(years=5))
                      & (panel["date"] <= embargo_end)]
        max_train = train["date"].max()
        gap = len(pd.bdate_range(max_train, date)) - 1
        seen.append(gap)
        assert gap >= TARGET_HORIZON, (
            f"embargo too short at {date.date()}: last training row "
            f"{max_train.date()} is {gap} business days back, target needs "
            f">= {TARGET_HORIZON}")
    assert seen, "no walk-forward steps exercised"
    assert min(seen) >= TARGET_HORIZON


def test_walk_forward_with_short_embargo_is_detectably_leaky():
    """The embargo parameter must actually move the training cutoff.

    With embargo_days=1 the last training row sits inside the 21-day target
    window, so its label overlaps the test period -- the exact leak that
    inflated the GBM Sharpe to 1.84.
    """
    panel = _panel(n_days=1600, n_tickers=6)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    date = dates[-1]
    leaky_end = date - pd.offsets.BDay(1)
    safe_end = date - pd.offsets.BDay(22)
    leaky_gap = len(pd.bdate_range(
        panel[panel["date"] <= leaky_end]["date"].max(), date)) - 1
    safe_gap = len(pd.bdate_range(
        panel[panel["date"] <= safe_end]["date"].max(), date)) - 1
    assert leaky_gap < TARGET_HORIZON <= safe_gap


def test_cross_sectional_sort_has_no_lookahead():
    """A factor that perfectly knows the CURRENT month's return must earn
    nothing, because the harness may only trade on the PRIOR month's factor.

    If someone removes the shift(1), this test reports a huge Sharpe.
    """
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2010-01-01", periods=1500)
    tickers = [f"T{i}" for i in range(20)]
    rets = pd.DataFrame(rng.standard_normal((len(dates), len(tickers))) * 0.01,
                        index=dates, columns=tickers)
    prices = 100 * np.exp(rets.cumsum())

    # oracle factor: this month's realized return, stamped inside the month
    monthly = np.log(prices).diff().resample("ME").sum()
    oracle = monthly.reindex(prices.index, method="bfill")

    res = cross_sectional_sort_backtest(oracle, prices, n_quantiles=5,
                                        min_names_per_leg=4)
    sharpe = res.stats()["sharpe"]
    assert abs(sharpe) < 1.5, (
        f"oracle factor earned Sharpe {sharpe:.2f} -- the one-period lag in "
        f"cross_sectional_sort_backtest is not being applied")


def test_stationary_bootstrap_block_one_is_iid():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 1.0, rng)
    assert len(idx) == 500 and idx.min() >= 0 and idx.max() < 500
    # consecutive-index runs should be rare under iid
    runs = (np.diff(idx) == 1).mean()
    assert runs < 0.05


def test_stationary_bootstrap_preserves_dependence():
    """With expected_block=20 the resample should retain autocorrelation
    that an iid resample destroys."""
    n = 4000
    x = pd.Series(np.random.default_rng(1).standard_normal(n)).rolling(10).mean().dropna().values
    rng = np.random.default_rng(2)
    ac_block = np.mean([pd.Series(x[stationary_bootstrap_indices(len(x), 20, rng)]).autocorr(1)
                        for _ in range(20)])
    ac_iid = np.mean([pd.Series(x[stationary_bootstrap_indices(len(x), 1, rng)]).autocorr(1)
                      for _ in range(20)])
    assert ac_block > 0.5, f"block bootstrap lost dependence: ac1={ac_block:.2f}"
    assert abs(ac_iid) < 0.1, f"iid bootstrap retained dependence: ac1={ac_iid:.2f}"


def test_iid_bootstrap_understates_ci_on_overlapping_returns():
    """The documented hazard: overlapping k-period returns are serially
    dependent, so an iid bootstrap reports a CI that is too NARROW."""
    rng = np.random.default_rng(5)
    daily = rng.standard_normal(4000) * 0.01 + 0.0004
    overlapping = pd.Series(daily).rolling(21).sum().dropna().values
    iid = sharpe_ci(overlapping, 252, n_boot=1500, seed=0, expected_block=1)
    blk = sharpe_ci(overlapping, 252, n_boot=1500, seed=0, expected_block=21)
    assert (blk["hi"] - blk["lo"]) > 1.5 * (iid["hi"] - iid["lo"]), (
        f"block CI {blk['hi'] - blk['lo']:.2f} should be much wider than "
        f"iid CI {iid['hi'] - iid['lo']:.2f} on overlapping returns")


def test_paired_diff_ci_tighter_than_treating_series_as_independent():
    """Two highly-correlated strategies: the paired bootstrap must be used."""
    rng = np.random.default_rng(7)
    bh = rng.standard_normal(240) * 0.04 + 0.006
    gate = np.where(rng.random(240) > 0.5, bh, 0.0)
    d = paired_sharpe_diff_ci(gate, bh, 12, n_boot=2000, seed=0)
    assert d["n"] == 240
    assert d["lo"] < d["point"] < d["hi"]
