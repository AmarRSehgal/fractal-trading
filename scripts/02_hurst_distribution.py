"""Compute DFA Hurst on a stock universe and report the distribution.

This is the diagnostic that tells you whether there is enough cross-sectional
dispersion in H to build a strategy (Tier 1 idea #1). If H across stocks is
tightly clustered near 0.5 with narrow CI, the whole thesis is dead.

Usage:
    python3 scripts/02_hurst_distribution.py [--universe sp100|dow30]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fractal_trading.data import load_prices, log_returns
from fractal_trading.hurst import dfa, modified_rs
from fractal_trading.universe import dow30_tickers, sp100_tickers


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="sp100", choices=["dow30", "sp100"])
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    tickers = dow30_tickers() if args.universe == "dow30" else sp100_tickers()
    print(f"Loading {len(tickers)} tickers from {args.start}...")
    prices = load_prices(tickers, start=args.start, end=args.end)
    print(f"  {prices.shape[0]} days x {prices.shape[1]} tickers")

    rets = log_returns(prices).dropna(how="all")

    # Full-sample Hurst per ticker
    records = []
    for tkr in prices.columns:
        series = rets[tkr].dropna().values
        if len(series) < 500:
            continue
        H_dfa = dfa(series)
        rs = modified_rs(series)
        records.append({
            "ticker": tkr,
            "n_days": len(series),
            "H_dfa": H_dfa,
            "H_rs_implied": rs["H_implied"],
            "rs_rejects_null": rs["rejects_null_95"],
            "ann_vol": series.std() * np.sqrt(252),
        })

    df = pd.DataFrame(records).sort_values("H_dfa")
    out_csv = RESULTS / f"hurst_distribution_{args.universe}.csv"
    df.to_csv(out_csv, index=False)

    print()
    print(f"Hurst distribution across {len(df)} stocks ({args.universe}):")
    print(f"  DFA H:          mean={df['H_dfa'].mean():.3f}  std={df['H_dfa'].std():.3f}")
    print(f"                  p05={df['H_dfa'].quantile(0.05):.3f}  p95={df['H_dfa'].quantile(0.95):.3f}")
    print(f"  Mod R/S rejections: {df['rs_rejects_null'].sum()} / {len(df)}")
    print()
    print("Top 5 most PERSISTENT (highest DFA H):")
    print(df.nlargest(5, "H_dfa")[["ticker", "H_dfa", "H_rs_implied", "rs_rejects_null"]].to_string(index=False))
    print()
    print("Top 5 most ANTI-PERSISTENT (lowest DFA H):")
    print(df.nsmallest(5, "H_dfa")[["ticker", "H_dfa", "H_rs_implied", "rs_rejects_null"]].to_string(index=False))
    print(f"\nSaved: {out_csv}")

    # Simple plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["H_dfa"].dropna(), bins=25, edgecolor="black")
        ax.axvline(0.5, color="red", linestyle="--", label="H=0.5 (random walk)")
        ax.set_xlabel("DFA Hurst exponent")
        ax.set_ylabel("Count")
        ax.set_title(f"Hurst distribution across {args.universe.upper()} ({args.start} to today)")
        ax.legend()
        fig.tight_layout()
        png = RESULTS / f"hurst_distribution_{args.universe}.png"
        fig.savefig(png, dpi=120)
        print(f"Saved: {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
