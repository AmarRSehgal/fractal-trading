"""ETF-level Hurst sort: last remaining directional fractal idea.

Hypothesis: a curated cross-asset ETF universe (equity sectors, countries,
bonds, commodities, FX) should show wider H dispersion than individual US
stocks, because aggregating instruments and mixing asset classes creates
more heterogeneity. If any cross-sectional H edge exists in retail
instruments, it should be here.

Diagnostics:
  - H distribution with asset-class color-coding
  - Per-leg asset-class composition at each rebalance
  - Check whether the sort is "Hurst" or "asset class in disguise"

Usage:
    python3 scripts/08_etf_hurst_sort.py
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fractal_trading.backtest import cross_sectional_sort_backtest, rolling_factor
from fractal_trading.data import load_prices
from fractal_trading.hurst import dfa, modified_rs
from fractal_trading.universe import etf_universe


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--n_quantiles", type=int, default=5)
    parser.add_argument("--cost_bps", type=float, default=10.0)
    args = parser.parse_args()

    univ = etf_universe()
    tickers = sorted(univ.keys())
    print(f"Loading {len(tickers)} ETFs from {args.start}...")
    prices = load_prices(tickers, start=args.start, end=args.end)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.6))
    print(f"  after coverage filter: {prices.shape}")

    # === H distribution diagnostic, with asset class tagging ===
    rets = np.log(prices).diff().dropna(how="all")
    records = []
    for tkr in prices.columns:
        s = rets[tkr].dropna().values
        if len(s) < 500:
            continue
        H = dfa(s)
        rs = modified_rs(s)
        records.append({
            "ticker": tkr,
            "asset_class": univ.get(tkr, "unknown"),
            "H_dfa": H,
            "H_rs_implied": rs["H_implied"],
            "rs_rejects_null": rs["rejects_null_95"],
            "ann_vol": float(s.std() * np.sqrt(252)),
            "n_days": len(s),
        })
    h_df = pd.DataFrame(records).sort_values("H_dfa")
    h_df.to_csv(RESULTS / "hurst_distribution_etf.csv", index=False)

    print()
    print(f"=== ETF full-sample Hurst distribution ({len(h_df)} tickers) ===")
    print(f"  mean={h_df['H_dfa'].mean():.3f}  std={h_df['H_dfa'].std():.3f}")
    print(f"  p05={h_df['H_dfa'].quantile(0.05):.3f}  p95={h_df['H_dfa'].quantile(0.95):.3f}")
    print(f"  range={h_df['H_dfa'].max() - h_df['H_dfa'].min():.3f}")
    print(f"  rejects Lo null at 95%: {h_df['rs_rejects_null'].sum()} / {len(h_df)}")

    print("\nTop 5 most PERSISTENT:")
    print(h_df.nlargest(5, "H_dfa")[["ticker", "asset_class", "H_dfa", "H_rs_implied", "rs_rejects_null"]].to_string(index=False))
    print("\nTop 5 most ANTI-PERSISTENT:")
    print(h_df.nsmallest(5, "H_dfa")[["ticker", "asset_class", "H_dfa", "H_rs_implied", "rs_rejects_null"]].to_string(index=False))

    # Mean H by broad asset class
    print("\nMean full-sample DFA H by broad asset class:")
    h_df["broad_class"] = h_df["asset_class"].str.split("_").str[0]
    by_class = h_df.groupby("broad_class").agg(
        n=("ticker", "count"), H_mean=("H_dfa", "mean"), H_std=("H_dfa", "std"),
    )
    print(by_class.to_string())

    # === Rolling factor + backtest ===
    print(f"\nComputing rolling DFA Hurst (lookback={args.lookback}d, step=21d)...")
    t0 = time.time()
    factor = rolling_factor(prices, dfa, lookback_days=args.lookback, step_days=21)
    print(f"  {time.time() - t0:.1f}s; factor shape {factor.shape}")

    print(f"Running quintile sort backtest...")
    result = cross_sectional_sort_backtest(
        factor, prices, n_quantiles=args.n_quantiles, min_names_per_leg=5,
    )

    print()
    print(result.report(
        title=f"ETF Hurst sort L/S (lb={args.lookback}d, Q{args.n_quantiles})",
        cost_bps_per_side=args.cost_bps,
    ))

    # === Composition analysis: what does the long and short leg actually hold? ===
    if not result.holdings.empty:
        # For each rebal date, record long picks (weight > 0) and short picks (< 0)
        long_counts = Counter()
        short_counts = Counter()
        long_class_counts = Counter()
        short_class_counts = Counter()
        n_rebals = 0
        for date, row in result.holdings.iterrows():
            longs = row[row > 0].index
            shorts = row[row < 0].index
            if len(longs) == 0 and len(shorts) == 0:
                continue
            n_rebals += 1
            for t in longs:
                long_counts[t] += 1
                long_class_counts[univ.get(t, "unknown")] += 1
            for t in shorts:
                short_counts[t] += 1
                short_class_counts[univ.get(t, "unknown")] += 1

        print(f"\n=== Composition over {n_rebals} rebalances ===")
        print("Most frequent LONG tickers:")
        for tkr, c in long_counts.most_common(10):
            print(f"  {tkr:<6s} {univ.get(tkr, '?'):<25s} {c}/{n_rebals} ({100*c/n_rebals:.0f}%)")
        print("Most frequent SHORT tickers:")
        for tkr, c in short_counts.most_common(10):
            print(f"  {tkr:<6s} {univ.get(tkr, '?'):<25s} {c}/{n_rebals} ({100*c/n_rebals:.0f}%)")

        # Broad class breakdown
        print("\nLong leg asset-class mix (% of total long positions):")
        total_long = sum(long_class_counts.values())
        for cls, c in sorted(long_class_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cls:<28s} {100*c/total_long:.1f}%")
        print("Short leg asset-class mix:")
        total_short = sum(short_class_counts.values())
        for cls, c in sorted(short_class_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cls:<28s} {100*c/total_short:.1f}%")

    # Save outputs
    result.portfolio_returns.to_csv(RESULTS / "hurst_ls_returns_etf.csv", header=["return"])
    stats_gross = result.stats(cost_bps_per_side=0.0)
    stats_net = result.stats(cost_bps_per_side=args.cost_bps)
    lo_g, pt_g, hi_g = result.bootstrap_sharpe_ci(0.0)
    lo_n, pt_n, hi_n = result.bootstrap_sharpe_ci(args.cost_bps)
    summary = pd.DataFrame({"gross": stats_gross, "net": stats_net})
    summary.loc["sharpe_ci_low"] = [lo_g, lo_n]
    summary.loc["sharpe_ci_high"] = [hi_g, hi_n]
    summary.to_csv(RESULTS / "hurst_sort_etf_stats.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # H distribution by broad class
        broad_classes = h_df["broad_class"].unique()
        for bc in sorted(broad_classes):
            vals = h_df[h_df["broad_class"] == bc]["H_dfa"]
            axes[0, 0].scatter(vals, [bc] * len(vals), alpha=0.7)
        axes[0, 0].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="H=0.5")
        axes[0, 0].set_xlabel("DFA H (full-sample)")
        axes[0, 0].set_title(f"ETF Hurst by broad asset class (n={len(h_df)})")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis="x")

        axes[0, 1].hist(h_df["H_dfa"], bins=25, edgecolor="black")
        axes[0, 1].axvline(0.5, color="red", linestyle="--", label="H=0.5")
        axes[0, 1].set_title(f"ETF Hurst distribution  mean={h_df['H_dfa'].mean():.3f}")
        axes[0, 1].set_xlabel("DFA H"); axes[0, 1].legend()

        # Equity curves
        cum_g = (1 + result.portfolio_returns).cumprod()
        cum_n = (1 + result.net_returns(args.cost_bps)).cumprod()
        axes[1, 0].plot(cum_g.index, cum_g.values, label=f"gross (Sharpe {stats_gross['sharpe']:.2f})")
        axes[1, 0].plot(cum_n.index, cum_n.values, label=f"net 10bps (Sharpe {stats_net['sharpe']:.2f})")
        axes[1, 0].set_title("ETF Hurst L/S equity")
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

        # Monthly returns scatter
        axes[1, 1].plot(result.portfolio_returns.index, result.portfolio_returns.values,
                         marker="o", ms=3, linestyle="None")
        axes[1, 1].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("Monthly L/S returns")
        axes[1, 1].grid(alpha=0.3)

        fig.tight_layout()
        png = RESULTS / "hurst_sort_etf.png"
        fig.savefig(png, dpi=120)
        print(f"\nSaved: {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
