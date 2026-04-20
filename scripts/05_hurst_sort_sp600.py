"""Tier 1 idea #1 on small-caps: Hurst sort on S&P 600.

Hypothesis: small-caps should have wider H dispersion than S&P 100 (our
earlier null), making the sort more tradable. Reports gross AND net-of-cost
stats and bootstrap Sharpe CI.

Usage:
    python3 scripts/05_hurst_sort_sp600.py [--lookback 500] [--cost_bps 5]
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
from fractal_trading.universe import sp600_tickers


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--n_quantiles", type=int, default=5)
    parser.add_argument("--cost_bps", type=float, default=10.0,
                        help="one-way bps; applied to turnover")
    args = parser.parse_args()

    tickers = sp600_tickers()
    print(f"Loading {len(tickers)} S&P 600 tickers {args.start}...")
    prices = load_prices(tickers, start=args.start, end=args.end)
    # require >= 60% data coverage (small caps have delistings; relaxed filter)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.6))
    print(f"  after filter: {prices.shape}")

    # === Hurst distribution diagnostic on this universe ===
    rets = np.log(prices).diff().dropna(how="all")
    h_list = []
    for tkr in prices.columns:
        s = rets[tkr].dropna().values
        if len(s) >= 500:
            h_list.append((tkr, dfa(s)))
    h_df = pd.DataFrame(h_list, columns=["ticker", "H_dfa"]).dropna()
    print()
    print(f"=== S&P 600 full-sample Hurst distribution ({len(h_df)} stocks) ===")
    print(f"  mean={h_df['H_dfa'].mean():.3f}  std={h_df['H_dfa'].std():.3f}")
    print(f"  p05={h_df['H_dfa'].quantile(0.05):.3f}  p95={h_df['H_dfa'].quantile(0.95):.3f}")
    print(f"  range={h_df['H_dfa'].max() - h_df['H_dfa'].min():.3f}")
    h_df.to_csv(RESULTS / "hurst_distribution_sp600.csv", index=False)

    # === Rolling factor and backtest ===
    print(f"\nComputing rolling DFA Hurst (lookback={args.lookback}d, step=21d)...")
    t0 = time.time()
    factor = rolling_factor(prices, dfa, lookback_days=args.lookback, step_days=21)
    print(f"  done in {time.time() - t0:.1f}s; factor shape {factor.shape}")

    print(f"Running quintile sort backtest...")
    result = cross_sectional_sort_backtest(
        factor, prices, n_quantiles=args.n_quantiles, min_names_per_leg=20,
    )

    print()
    print(result.report(
        title=f"Hurst sort L/S on S&P 600 (lb={args.lookback}d)",
        cost_bps_per_side=args.cost_bps,
    ))

    # Save
    result.portfolio_returns.to_csv(RESULTS / "hurst_ls_returns_sp600.csv", header=["return"])
    stats_gross = result.stats(cost_bps_per_side=0.0)
    stats_net = result.stats(cost_bps_per_side=args.cost_bps)
    lo_g, pt_g, hi_g = result.bootstrap_sharpe_ci(0.0)
    lo_n, pt_n, hi_n = result.bootstrap_sharpe_ci(args.cost_bps)
    summary = pd.DataFrame({
        "gross": stats_gross,
        "net": stats_net,
    })
    summary.loc["sharpe_ci_low"] = [lo_g, lo_n]
    summary.loc["sharpe_ci_high"] = [hi_g, hi_n]
    summary.to_csv(RESULTS / "hurst_sort_sp600_stats.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

        axes[0].hist(h_df["H_dfa"], bins=40, edgecolor="black")
        axes[0].axvline(0.5, color="red", linestyle="--", label="H=0.5")
        axes[0].set_title(f"S&P 600 Hurst distribution  (n={len(h_df)})")
        axes[0].set_xlabel("DFA Hurst"); axes[0].legend()

        cum_g = (1 + result.portfolio_returns).cumprod()
        cum_n = (1 + result.net_returns(args.cost_bps)).cumprod()
        axes[1].plot(cum_g.index, cum_g.values, label=f"gross (Sharpe {stats_gross['sharpe']:.2f})")
        axes[1].plot(cum_n.index, cum_n.values, label=f"net, {args.cost_bps:.0f}bps (Sharpe {stats_net['sharpe']:.2f})")
        axes[1].set_title("Hurst L/S equity (S&P 600)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        png = RESULTS / "hurst_sort_sp600.png"
        fig.savefig(png, dpi=120)
        print(f"Saved: {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
