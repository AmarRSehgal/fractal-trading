"""Tier 1 idea #1: cross-sectional Hurst sort backtest.

Sort a stock universe by rolling DFA Hurst. Long top quintile, short bottom.
Monthly rebalance. Report Sharpe, turnover, drawdown.

IMPORTANT CAVEATS:
- Universe is survivorship-biased (current S&P 100 only).
- No transaction costs modeled yet.
- Equal-weighted legs.

Treat any positive result with suspicion until you've re-run with a
bias-free universe and realistic costs.

Usage:
    python3 scripts/03_hurst_sort_backtest.py [--lookback 500]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fractal_trading.backtest import cross_sectional_sort_backtest, rolling_factor
from fractal_trading.data import load_prices
from fractal_trading.hurst import dfa
from fractal_trading.universe import dow30_tickers, sp100_tickers


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="sp100", choices=["dow30", "sp100"])
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--n_quantiles", type=int, default=5)
    parser.add_argument("--cost_bps", type=float, default=10.0)
    args = parser.parse_args()

    tickers = dow30_tickers() if args.universe == "dow30" else sp100_tickers()
    print(f"Loading {len(tickers)} tickers {args.start}...")
    prices = load_prices(tickers, start=args.start, end=args.end)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    print(f"  after filter: {prices.shape}")

    print(f"Computing rolling DFA Hurst (lookback={args.lookback}d, step=21d)...")
    t0 = time.time()
    factor = rolling_factor(prices, dfa, lookback_days=args.lookback, step_days=21)
    print(f"  done in {time.time() - t0:.1f}s; factor shape {factor.shape}")

    print(f"Running quintile sort backtest...")
    result = cross_sectional_sort_backtest(
        factor, prices, n_quantiles=args.n_quantiles, min_names_per_leg=5,
    )

    print()
    print(result.report(
        title=f"Hurst sort L/S ({args.universe}, top Q{args.n_quantiles} vs bottom)",
        cost_bps_per_side=args.cost_bps,
    ))
    stats = result.stats(cost_bps_per_side=args.cost_bps)

    # Save outputs
    result.portfolio_returns.to_csv(RESULTS / f"hurst_ls_returns_{args.universe}.csv", header=["return"])
    pd.Series(stats).to_csv(RESULTS / f"hurst_ls_stats_{args.universe}.csv", header=["value"])

    # Equity curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        r = result.portfolio_returns
        cum = (1 + r).cumprod()
        axes[0].plot(cum.index, cum.values, label="L/S")
        axes[0].plot((1 + result.long_returns).cumprod(), label="Long top-Q", alpha=0.6)
        axes[0].plot((1 + result.short_returns).cumprod(), label="Short bottom-Q", alpha=0.6)
        axes[0].set_ylabel("Cumulative return")
        axes[0].set_title(f"Hurst quintile sort: {args.universe.upper()} (survivorship biased)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(r.index, r.values, marker="o", ms=3, linestyle="None")
        axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Monthly L/S return")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        png = RESULTS / f"hurst_ls_equity_{args.universe}.png"
        fig.savefig(png, dpi=120)
        print(f"\nSaved: {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
