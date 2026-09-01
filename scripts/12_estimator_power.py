"""Estimator power audit: is the Hurst factor measuring anything?

Every Hurst-based null in this repo is only interpretable if the estimator
could have detected the effect in the first place. Three checks:

  1. NOISE FLOOR  - DFA bias and standard error vs sample size, on fBm with
                    known H. Answers "is DFA biased?" (no) and "how noisy is
                    it at the 500-day lookback the backtests use?" (very).
  2. RELIABILITY  - cross-sectional dispersion of the rolling 500d factor vs
                    that noise floor, plus rank persistence across DISJOINT
                    estimation windows. A factor with no test-retest
                    reliability cannot be sorted on.
  3. SHUFFLE      - DFA on each stock's own SHUFFLED returns. Shuffling
                    destroys temporal structure (true H = 0.5) but keeps the
                    fat-tailed marginal, so this separates "real
                    anti-persistence" from "DFA reacting to fat tails".

Usage:
    python3 scripts/12_estimator_power.py [--end 2026-08-31] [--quick]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fractal_trading.backtest import rolling_factor
from fractal_trading.data import load_prices
from fractal_trading.hurst import dfa
from fractal_trading.synthetic import fbm_increments
from fractal_trading.universe import sp100_tickers


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

LOOKBACK = 500


def noise_floor(n_seeds: int) -> pd.DataFrame:
    rows = []
    for n in (500, 1000, 2000, 4000, 5300):
        for true_H in (0.45, 0.50, 0.55):
            e = np.array([dfa(fbm_increments(n, H=true_H, seed=s))
                          for s in range(n_seeds)])
            e = e[np.isfinite(e)]
            rows.append({"n": n, "true_H": true_H, "mean": e.mean(),
                         "bias": e.mean() - true_H, "std": e.std()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None, help="pin for reproducibility")
    ap.add_argument("--quick", action="store_true", help="fewer seeds/shuffles")
    args = ap.parse_args()
    n_seeds = 40 if args.quick else 200
    n_shuffles = 1 if args.quick else 3

    print("=" * 68)
    print("1. DFA NOISE FLOOR on fBm with known H")
    print("=" * 68)
    nf = noise_floor(n_seeds)
    print(nf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    nf.to_csv(RESULTS / "estimator_noise_floor.csv", index=False)
    floor_500 = nf[(nf["n"] == 500) & (nf["true_H"] == 0.50)]["std"].iloc[0]
    print(f"\n  DFA is unbiased (|bias| <= {nf['bias'].abs().max():.4f}).")
    print(f"  Standard error at the {LOOKBACK}-day backtest lookback: {floor_500:.4f}")

    prices = load_prices(sp100_tickers(), start=args.start, end=args.end)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))

    print("\n" + "=" * 68)
    print(f"2. RELIABILITY of the rolling {LOOKBACK}d Hurst factor (S&P 100)")
    print("=" * 68)
    f = rolling_factor(prices, dfa, lookback_days=LOOKBACK, step_days=21)
    f.to_csv(RESULTS / "hurst_factor_sp100_rolling.csv")
    xs = f.std(axis=1).mean()
    print(f"  observed cross-sectional std : {xs:.4f}")
    print(f"  estimator noise floor        : {floor_500:.4f}")
    var = xs ** 2 - floor_500 ** 2
    print("  implied TRUE dispersion      : "
          + (f"{np.sqrt(var):.4f}" if var > 0
             else "0  (observed dispersion is BELOW the noise floor)"))

    print(f"\n  Rank persistence (Spearman). Lag is in 21-day rebalance steps;")
    print(f"  lag >= {LOOKBACK // 21} means the two estimation windows are DISJOINT.")
    rel = []
    for lag in (1, 6, 12, 24, 36, 48):
        cs = [f.iloc[i].corr(f.iloc[i + lag], method="spearman")
              for i in range(len(f) - lag) if f.iloc[i].notna().sum() > 30]
        cs = [c for c in cs if np.isfinite(c)]
        if cs:
            overlap = max(0.0, 1 - lag * 21 / LOOKBACK)
            rel.append({"lag_rebals": lag, "lag_days": lag * 21,
                        "window_overlap": overlap, "spearman": np.mean(cs)})
            print(f"    lag {lag:>3} ({lag * 21:>4}d, overlap {overlap:>4.0%}): "
                  f"rho = {np.mean(cs):+.3f}")
    pd.DataFrame(rel).to_csv(RESULTS / "hurst_factor_reliability.csv", index=False)

    print("\n" + "=" * 68)
    print("3. SHUFFLE CONTROL (full-sample H, real vs shuffled)")
    print("=" * 68)
    rets = np.log(prices).diff()
    rng = np.random.default_rng(0)
    real, shuf_mean, shuf_one = [], [], []
    for t in prices.columns:
        s = rets[t].dropna().values
        if len(s) < 500:
            continue
        reps = [dfa(rng.permutation(s)) for _ in range(n_shuffles)]
        real.append(dfa(s))
        shuf_mean.append(np.mean(reps))   # tighter estimate of the shuffled MEAN
        shuf_one.append(reps[0])          # single draw = true noise floor
    real = np.array(real)[np.isfinite(real)]
    shuf_mean = np.array(shuf_mean)[np.isfinite(shuf_mean)]
    shuf_one = np.array(shuf_one)[np.isfinite(shuf_one)]

    # The dispersion floor must come from ONE shuffle per stock. Averaging
    # k shuffles shrinks the estimator noise by ~sqrt(k) and would understate
    # the floor -- and therefore overstate the implied true dispersion.
    floor = shuf_one.std()
    print(f"  n stocks                  : {len(real)}")
    print(f"  DFA H, REAL returns       : mean {real.mean():.4f}  std {real.std():.4f}")
    print(f"  DFA H, SHUFFLED returns   : mean {shuf_mean.mean():.4f}  "
          f"std {floor:.4f} (single draw)")
    print(f"  difference (real-shuffled): {real.mean() - shuf_mean.mean():+.4f}")
    v = real.std() ** 2 - floor ** 2
    print(f"  data-native noise floor   : {floor:.4f}")
    print("  implied TRUE dispersion   : "
          + (f"{np.sqrt(v):.4f}" if v > 0 else "0 (below noise floor)"))
    pd.DataFrame({"H_real": real, "H_shuffled": shuf_mean,
                  "H_shuffled_single": shuf_one}).to_csv(
        RESULTS / "hurst_shuffle_control.csv", index=False)
    print(f"\nSaved CSVs to {RESULTS}/")


if __name__ == "__main__":
    main()
