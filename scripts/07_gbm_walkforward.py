"""S2 proper: fractional-diff + momentum + vol as features in a GBM model
with walk-forward CV.

This is the fair test of López de Prado's proposal: FFD as ONE feature in
a nonlinear model, not a standalone cross-sectional z-score.

Features per stock, monthly:
  - frac_diff(log_price, d=0.4) z-score over 252d
  - 12-1 momentum (standard log-return momentum)
  - 1-month log return (short-term reversal candidate)
  - 60-day realized volatility of log returns
  - 252-day realized volatility

Model: LightGBM regressor predicting next-month log return.

Walk-forward: train on trailing 5 years, predict next month, step.

Compare:
  - Predictions-based long-short (long top decile of predicted, short bottom)
  - Vs 12-1 momentum alone as baseline

Usage:
    python3 scripts/07_gbm_walkforward.py [--end 2026-08-31] [--cost_bps 10]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_LGB = False

from fractal_trading.backtest import cross_sectional_sort_backtest
from fractal_trading.data import load_prices
from fractal_trading.fracdiff import frac_diff_ffd
from fractal_trading.universe import sp100_tickers


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def build_features(prices: pd.DataFrame, d: float = 0.4) -> dict:
    """Return a dict of feature DataFrames (dates x tickers)."""
    log_p = np.log(prices)
    rets = log_p.diff()

    features = {}

    # Momentum 12-1
    features["mom_12_1"] = (log_p - log_p.shift(252)) - (log_p - log_p.shift(21))

    # Short-term reversal (-1m return)
    features["ret_1m"] = log_p - log_p.shift(21)

    # Realized vols
    features["vol_60"] = rets.rolling(60).std() * np.sqrt(252)
    features["vol_252"] = rets.rolling(252).std() * np.sqrt(252)

    # Frac-diff z-score
    fd_df = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        s = log_p[col].dropna().values
        if len(s) < 500:
            continue
        fd = frac_diff_ffd(s, d=d)
        fd_series = pd.Series(fd, index=log_p[col].dropna().index)
        z = (fd_series - fd_series.rolling(252).mean()) / fd_series.rolling(252).std()
        fd_df[col] = z.reindex(prices.index)
    features["fd_z"] = fd_df

    return features


def make_panel(features: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """Convert feature dict into a long-format panel (date, ticker, features, target)."""
    log_p = np.log(prices)
    target = (log_p.shift(-21) - log_p)  # next-month log return

    frames = []
    for name, df in features.items():
        s = df.stack()
        s.name = name
        frames.append(s)
    frames.append(target.stack().rename("target"))

    panel = pd.concat(frames, axis=1, join="inner").dropna()
    panel.index.names = ["date", "ticker"]
    return panel.reset_index()


def walk_forward_predict(
    panel: pd.DataFrame,
    feature_cols: list,
    train_years: int = 5,
    embargo_days: int = 22,
    rebal_freq: str = "ME",
) -> pd.DataFrame:
    """Train/predict monthly, walking forward.

    embargo_days: drop training rows within this many business days of the
    test date. Required because our target is forward-21-day return: a
    training row at d_train has target spanning [d_train, d_train + 21].
    Without embargo, training rows with d_train > date - 21 use returns
    not yet observed at test time -> lookahead. Set embargo_days = 22 to
    safely exclude.
    """
    panel = panel.sort_values("date").copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # rebalance dates = last business day per month
    dates = panel["date"].sort_values().unique()
    dates_series = pd.Series(dates)
    month_ends = dates_series[dates_series.dt.is_month_end | (dates_series.shift(-1).dt.month != dates_series.dt.month)]
    month_ends = pd.DatetimeIndex(sorted(set(month_ends.dropna())))

    predictions = []
    for i, date in enumerate(month_ends):
        train_start = date - pd.DateOffset(years=train_years)
        embargo_end = date - pd.offsets.BDay(embargo_days)
        train = panel[(panel["date"] >= train_start) & (panel["date"] <= embargo_end)]
        test = panel[panel["date"] == date]
        if len(train) < 1000 or len(test) < 10:
            continue

        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_test = test[feature_cols].values

        if USE_LGB:
            model = lgb.LGBMRegressor(
                num_leaves=15, learning_rate=0.05, n_estimators=100,
                min_child_samples=50, verbose=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(max_depth=3, learning_rate=0.05, n_estimators=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        for t, p in zip(test["ticker"].values, preds):
            predictions.append({"date": date, "ticker": t, "pred": float(p)})

    return pd.DataFrame(predictions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None, help="pin for reproducibility")
    ap.add_argument("--d", type=float, default=0.4)
    ap.add_argument("--embargo_days", type=int, default=22,
                    help="must exceed the 21-day target horizon; see tests/test_embargo.py")
    ap.add_argument("--cost_bps", type=float, default=10.0)
    args = ap.parse_args()

    tickers = sp100_tickers()
    print(f"Loading {len(tickers)} S&P 100 tickers from {args.start}...")
    prices = load_prices(tickers, start=args.start, end=args.end)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    print(f"  after filter: {prices.shape}")

    print("Building features...")
    t0 = time.time()
    features = build_features(prices, d=args.d)
    print(f"  {time.time() - t0:.1f}s")

    print("Assembling panel...")
    panel = make_panel(features, prices)
    print(f"  panel rows: {len(panel):,}, unique dates: {panel['date'].nunique()}")

    feature_cols = ["mom_12_1", "ret_1m", "vol_60", "vol_252", "fd_z"]
    lib = "lightgbm" if USE_LGB else "sklearn GBM"
    print(f"Walk-forward training with {lib}...")
    t0 = time.time()
    preds = walk_forward_predict(panel, feature_cols, train_years=5,
                                 embargo_days=args.embargo_days)
    print(f"  {time.time() - t0:.1f}s, {len(preds)} predictions")

    if preds.empty:
        print("No predictions; aborting.")
        return

    # Build factor DataFrame from predictions (dates x tickers)
    factor_gbm = preds.pivot(index="date", columns="ticker", values="pred").reindex(prices.index).ffill()

    # Baseline: 12-1 momentum
    factor_mom = features["mom_12_1"]

    # Run both backtests
    print("\nBacktesting GBM predictions...")
    r_gbm = cross_sectional_sort_backtest(factor_gbm, prices, n_quantiles=5, min_names_per_leg=5)
    print(r_gbm.report(title="GBM walk-forward L/S", cost_bps_per_side=args.cost_bps))

    print("\nBacktesting 12-1 momentum baseline...")
    r_mom = cross_sectional_sort_backtest(factor_mom, prices, n_quantiles=5, min_names_per_leg=5)
    print(r_mom.report(title="Momentum 12-1 L/S (baseline)", cost_bps_per_side=args.cost_bps))

    # Feature importance diagnostic: refit on full panel to extract
    if USE_LGB and len(panel) > 1000:
        full_model = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=200, verbose=-1)
        full_model.fit(panel[feature_cols].values, panel["target"].values)
        importances = dict(zip(feature_cols, full_model.feature_importances_))
        print("\nFull-sample feature importance (informational; not used for prediction):")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {k:<12s} {v}")

    # Save
    preds.to_csv(RESULTS / "gbm_walkforward_predictions.csv", index=False)
    r_gbm.portfolio_returns.to_csv(RESULTS / "gbm_ls_returns.csv", header=["return"])
    r_mom.portfolio_returns.to_csv(RESULTS / "mom_baseline_returns.csv", header=["return"])

    summary = pd.DataFrame({
        "gbm": r_gbm.stats(cost_bps_per_side=args.cost_bps),
        "momentum_baseline": r_mom.stats(cost_bps_per_side=args.cost_bps),
    })
    summary.to_csv(RESULTS / "gbm_vs_mom_stats.csv")
    print(f"\nSaved results to {RESULTS}/")


if __name__ == "__main__":
    main()
