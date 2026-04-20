"""Residualized Hurst sort on ETFs.

The plain ETF Hurst sort (script 08) produced Sharpe -0.48 because H was
picking up asset-class fundamentals rather than fractal memory. Composition
showed longs dominated by commodities+EM, shorts by bonds+developed.

This script removes the asset-class and vol contamination:
  1. At each rebalance date, run OLS: H_i ~ C(asset_class_i) + vol_i
  2. Use the residual as the ranking signal
  3. Sort on residual, quintile L/S

If the residual has NO predictive power, we've confirmed that Hurst on
retail-accessible instruments carries no fractal alpha net of obvious
confounds. If the residual DOES have edge, we've isolated a real signal.

Usage:
    python3 scripts/09_residualized_hurst_etf.py
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

from fractal_trading.backtest import cross_sectional_sort_backtest, rolling_factor
from fractal_trading.data import load_prices
from fractal_trading.hurst import dfa
from fractal_trading.universe import etf_universe


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def broad_class(fine: str) -> str:
    """Take fine-grained class like 'commodity_oil' -> 'commodity',
    'intl_japan' -> 'intl', 'us_sector_tech' -> 'us_sector'."""
    parts = fine.split("_")
    if len(parts) >= 2 and parts[0] == "us" and parts[1] == "sector":
        return "us_sector"
    if len(parts) >= 2 and parts[0] == "us":
        return "us_" + parts[1]   # us_broad, us_style
    return parts[0] if parts else "unknown"


def residualize_factor(
    h_factor: pd.DataFrame,
    vol_factor: pd.DataFrame,
    asset_class_map: dict,
) -> pd.DataFrame:
    """At each rebal date, regress H_i = alpha + beta*vol_i + class dummies
    + residual. Uses BROAD asset class so classes have enough members."""
    dates = h_factor.index
    out = pd.DataFrame(index=dates, columns=h_factor.columns, dtype=float)

    for date in dates:
        h_row = h_factor.loc[date].dropna()
        v_row = vol_factor.loc[date].reindex(h_row.index)
        common = h_row.index.intersection(v_row.dropna().index)
        if len(common) < 20:
            continue
        df = pd.DataFrame({
            "H": h_row[common].values,
            "vol": v_row[common].values,
            "cls": [broad_class(asset_class_map.get(t, "unknown")) for t in common],
        }, index=common)
        # drop singleton broad classes (safety; should rarely trigger)
        class_counts = df["cls"].value_counts()
        keep_classes = class_counts[class_counts >= 2].index
        df = df[df["cls"].isin(keep_classes)]
        if len(df) < 20:
            continue
        try:
            model = ols("H ~ C(cls) + vol", data=df).fit()
            resid = pd.Series(model.resid.values, index=df.index)
            for tkr, val in resid.items():
                out.at[date, tkr] = val
        except Exception:
            continue
    return out.astype(float)


def ann_vol(returns_window: np.ndarray) -> float:
    s = returns_window[np.isfinite(returns_window)]
    if len(s) < 20:
        return np.nan
    return float(s.std() * np.sqrt(252))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--cost_bps", type=float, default=10.0)
    args = parser.parse_args()

    univ = etf_universe()
    tickers = sorted(univ.keys())
    print(f"Loading {len(tickers)} ETFs from {args.start}...")
    prices = load_prices(tickers, start=args.start, end=args.end)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.6))
    print(f"  after filter: {prices.shape}")

    # Rolling H and rolling annualized vol
    print(f"Computing rolling DFA Hurst (lookback={args.lookback}d)...")
    t0 = time.time()
    h_factor = rolling_factor(prices, dfa, lookback_days=args.lookback, step_days=21)
    print(f"  Hurst: {time.time() - t0:.1f}s")

    t0 = time.time()
    vol_factor = rolling_factor(prices, ann_vol, lookback_days=args.lookback, step_days=21)
    print(f"  Vol:   {time.time() - t0:.1f}s")

    # Residualize
    print("Residualizing H on asset class + vol...")
    resid_factor = residualize_factor(h_factor, vol_factor, univ)
    print(f"  residual factor shape: {resid_factor.shape}")

    # Quick diagnostic: how much does residualization reduce variance explained by class?
    diag_rows = []
    for date in h_factor.index[-20:]:  # last 20 rebals
        h_row = h_factor.loc[date].dropna()
        v_row = vol_factor.loc[date].reindex(h_row.index)
        df = pd.DataFrame({
            "H": h_row, "vol": v_row,
            "cls": [univ.get(t, "unknown") for t in h_row.index],
        }).dropna()
        if len(df) < 20:
            continue
        try:
            m = ols("H ~ C(cls) + vol", data=df).fit()
            diag_rows.append({"date": date, "r2": m.rsquared, "n": len(df)})
        except Exception:
            continue
    if diag_rows:
        d = pd.DataFrame(diag_rows)
        print(f"\nDiagnostic (last 20 rebals): R^2 of H ~ class + vol")
        print(f"  mean R^2 = {d['r2'].mean():.3f}  -- fraction of H variance explained by confounds")
        print(f"  residual factor captures what's LEFT after removing that.")

    # Run backtest on residuals
    print("\nRunning backtest on RESIDUAL factor...")
    result_resid = cross_sectional_sort_backtest(
        resid_factor, prices, n_quantiles=5, min_names_per_leg=5,
    )
    print(result_resid.report(
        title=f"Residualized Hurst sort L/S on ETFs (lb={args.lookback}d)",
        cost_bps_per_side=args.cost_bps,
    ))

    # Composition check
    if not result_resid.holdings.empty:
        long_class = Counter()
        short_class = Counter()
        for date, row in result_resid.holdings.iterrows():
            for t in row[row > 0].index:
                long_class[univ.get(t, "unknown")] += 1
            for t in row[row < 0].index:
                short_class[univ.get(t, "unknown")] += 1
        print("\nResidual-factor long leg asset-class mix (top 10):")
        total = sum(long_class.values())
        for cls, c in sorted(long_class.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cls:<28s} {100 * c / total:.1f}%")
        print("Residual-factor short leg asset-class mix (top 10):")
        total = sum(short_class.values())
        for cls, c in sorted(short_class.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cls:<28s} {100 * c / total:.1f}%")

    # Also report the plain H result for comparison (recomputing cheap since
    # we already have h_factor)
    print("\nFor comparison: plain H sort (NOT residualized)...")
    result_plain = cross_sectional_sort_backtest(
        h_factor, prices, n_quantiles=5, min_names_per_leg=5,
    )
    print(result_plain.report(
        title="Plain H sort (no residualization)",
        cost_bps_per_side=args.cost_bps,
    ))

    # Save
    result_resid.portfolio_returns.to_csv(
        RESULTS / "residualized_hurst_etf_returns.csv", header=["return"]
    )
    pd.DataFrame({
        "plain_gross": result_plain.stats(0.0),
        "plain_net": result_plain.stats(args.cost_bps),
        "resid_gross": result_resid.stats(0.0),
        "resid_net": result_resid.stats(args.cost_bps),
    }).to_csv(RESULTS / "residualized_hurst_etf_stats.csv")


if __name__ == "__main__":
    main()
