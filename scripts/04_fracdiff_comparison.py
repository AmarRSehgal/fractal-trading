"""Tier 1 idea #2: fractionally-differenced price as a momentum-style factor.

Head-to-head: classical 12-1 momentum vs a z-score of frac-diff(log price).
Same universe, same sort mechanics, same backtest harness.

Question: does frac-diff add information vs plain log-return momentum?

Usage:
    python3 scripts/04_fracdiff_comparison.py [--d 0.4]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fractal_trading.backtest import cross_sectional_sort_backtest
from fractal_trading.data import load_prices
from fractal_trading.fracdiff import frac_diff_ffd
from fractal_trading.universe import sp100_tickers


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """12-month minus 1-month log return, indexed by date."""
    log_p = np.log(prices)
    # 252 trading days ~ 12m, 21 ~ 1m
    r_12 = log_p - log_p.shift(252)
    r_1 = log_p - log_p.shift(21)
    return (r_12 - r_1)


def fracdiff_zscore_factor(prices: pd.DataFrame, d: float, window: int = 252) -> pd.DataFrame:
    """For each ticker: compute FFD of log price, then rolling z-score."""
    log_p = np.log(prices)
    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        series = log_p[col].dropna().values
        if len(series) < window + 100:
            continue
        fd = frac_diff_ffd(series, d=d)
        fd_series = pd.Series(fd, index=log_p[col].dropna().index)
        rolling_mean = fd_series.rolling(window).mean()
        rolling_std = fd_series.rolling(window).std()
        z = (fd_series - rolling_mean) / rolling_std
        out[col] = z.reindex(prices.index)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--d", type=float, default=0.4)
    parser.add_argument("--cost_bps", type=float, default=10.0)
    args = parser.parse_args()

    tickers = sp100_tickers()
    print(f"Loading {len(tickers)} tickers {args.start}...")
    prices = load_prices(tickers, start=args.start, end=args.end)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    print(f"  after filter: {prices.shape}")

    print("Computing momentum 12-1 factor...")
    t0 = time.time()
    mom = momentum_12_1(prices)
    print(f"  {time.time() - t0:.1f}s")

    print(f"Computing FFD z-score factor (d={args.d})...")
    t0 = time.time()
    fd = fracdiff_zscore_factor(prices, d=args.d)
    print(f"  {time.time() - t0:.1f}s")

    results = {}
    for name, factor in [("momentum_12_1", mom), ("fracdiff_z", fd)]:
        print(f"\nBacktesting {name}...")
        r = cross_sectional_sort_backtest(factor, prices, n_quantiles=5, min_names_per_leg=5)
        results[name] = r
        print(r.report(title=name, cost_bps_per_side=args.cost_bps))

    # Combined factor: average z-scores (normalize both first)
    mom_z = (mom - mom.rolling(252).mean()) / mom.rolling(252).std()
    combo = 0.5 * mom_z.fillna(0) + 0.5 * fd.fillna(0)
    combo = combo.where(mom_z.notna() & fd.notna())
    print("\nBacktesting combined (equal-weighted z-scores)...")
    r_combo = cross_sectional_sort_backtest(combo, prices, n_quantiles=5, min_names_per_leg=5)
    results["combined"] = r_combo
    print(r_combo.report(title="combined", cost_bps_per_side=args.cost_bps))

    # Comparison table
    print()
    print("=" * 72)
    print(f"{'Factor':<20s} {'Sharpe':>8s} {'AnnRet':>8s} {'AnnVol':>8s} {'MaxDD':>8s} {'Turnover':>10s}")
    print("-" * 72)
    for name, r in results.items():
        s = r.stats(cost_bps_per_side=args.cost_bps)
        print(f"{name:<20s} {s['sharpe']:>8.3f} {s['ann_return']:>8.3f} "
              f"{s['ann_vol']:>8.3f} {s['max_drawdown']:>8.3f} {s['turnover']:>10.3f}")
    print("=" * 72)

    # Save
    summary = pd.DataFrame(
        {name: r.stats(cost_bps_per_side=args.cost_bps) for name, r in results.items()}
    ).T
    summary.to_csv(RESULTS / "fracdiff_vs_momentum_stats.csv")
    for name, r in results.items():
        r.portfolio_returns.to_csv(RESULTS / f"fracdiff_compare_{name}_returns.csv", header=["return"])

    # Equity curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, r in results.items():
            cum = (1 + r.portfolio_returns).cumprod()
            ax.plot(cum.index, cum.values, label=name)
        ax.set_title(f"L/S equity curves: momentum vs frac-diff (d={args.d}) vs combined")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = RESULTS / "fracdiff_vs_momentum_equity.png"
        fig.savefig(png, dpi=120)
        print(f"Saved: {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
