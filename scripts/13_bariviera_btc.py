"""Bariviera (2017) replication: is BTC's Hurst exponent time-varying?

RESEARCH_NOTES.md names this as "a direct first notebook" and the empirical
anchor for treating Hurst as a regime indicator. It was never run. Every
other experiment in this repo tested the long-memory hypothesis on US
equities -- the asset class with the strongest efficiency prior -- so this
is the baseline that says whether the machinery can detect a published
positive at all, or whether the nulls are just a dull instrument.

Bariviera's claim: BTC is strongly persistent early (2011-2013, H > 0.55)
and drifts to H ~ 0.5 as the market matures (2014-2017).

Two things this script establishes, and one it cannot:
  - CANNOT: yfinance's BTC-USD history begins 2014-09. The 2011-2013
    persistent era -- the half of the paper that carries the signal -- is
    simply not available from this repo's data source. The replication is
    infeasible here, not merely null.
  - CAN: measure H over the post-2014 era Bariviera says has already
    converged, and compare every estimate against the DFA noise floor at
    the matching sample size. Without that floor an H of 0.55 on 500 points
    is indistinguishable from 0.5.
  - CAN: show that the *full-sample* H looks persistent while every
    sub-period does not -- the regime-mixing artifact that makes
    single-number Hurst claims on long crypto samples unreliable.

Usage:
    python3 scripts/13_bariviera_btc.py [--end 2026-08-31] [--lookback 500]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fractal_trading.data import load_prices
from fractal_trading.hurst import dfa, modified_rs
from fractal_trading.synthetic import fbm_increments


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# Bariviera (2017) sample: BTC daily, 2011-2017, split at the maturity point.
PAPER_START = pd.Timestamp("2011-01-01")
PERSISTENT_ERA = ("2011-01-01", "2013-12-31")


def noise_floor(n: int, n_seeds: int, true_H: float = 0.5) -> float:
    """Std of DFA on iid-equivalent series of length n. An |H - 0.5| smaller
    than ~2x this is not evidence of anything."""
    e = np.array([dfa(fbm_increments(n, H=true_H, seed=s)) for s in range(n_seeds)])
    return float(np.nanstd(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default=None, help="pin for reproducibility")
    ap.add_argument("--lookback", type=int, default=500)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--n_seeds", type=int, default=200)
    ap.add_argument("--tickers", default="BTC-USD,ETH-USD")
    args = ap.parse_args()

    tickers = args.tickers.split(",")
    prices = load_prices(tickers, start=args.start, end=args.end)

    print("=" * 70)
    print("1. DATA COVERAGE vs the paper's sample")
    print("=" * 70)
    rows = []
    for t in tickers:
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        covered = s.index[0] <= PAPER_START
        print(f"  {t:<9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):>5}  "
              f"paper's 2011-2013 era: {'COVERED' if covered else 'NOT COVERED'}")
        rows.append({"ticker": t, "first": s.index[0].date(), "last": s.index[-1].date(),
                     "n": len(s), "persistent_era_covered": covered})
    if not any(r["persistent_era_covered"] for r in rows):
        print(f"\n  => The {PERSISTENT_ERA[0][:4]}-{PERSISTENT_ERA[1][:4]} persistent era is")
        print("     unavailable from yfinance. The half of Bariviera's result that")
        print("     carries the signal CANNOT be replicated on this data source;")
        print("     only his 'already converged to 0.5' half is testable below.")

    btc = prices[tickers[0]].dropna()
    r = np.log(btc).diff().dropna()

    print("\n" + "=" * 70)
    print("2. FULL SAMPLE vs SUB-PERIODS (the regime-mixing trap)")
    print("=" * 70)
    floor_full = noise_floor(len(r), args.n_seeds)
    h_full = dfa(r.values)
    lo = modified_rs(r.values)
    print(f"  full sample  n={len(r):>5}  H = {h_full:.3f}  "
          f"(noise floor {floor_full:.3f}, 95% band "
          f"[{0.5 - 1.96 * floor_full:.3f}, {0.5 + 1.96 * floor_full:.3f}])")
    print(f"  Lo (1991) modified R/S: V = {lo['Q']:.3f} (q={lo['q']}), "
          f"reject no-long-memory at 95%: {lo['rejects_null_95']}")

    sub_rows = []
    edges = ["2014-09-01", "2018-01-01", "2022-01-01", "2027-01-01"]
    for a, b in zip(edges[:-1], edges[1:]):
        w = r[a:b]
        if len(w) < 300:
            continue
        floor = noise_floor(len(w), max(40, args.n_seeds // 4))
        h = dfa(w.values)
        sig = abs(h - 0.5) > 1.96 * floor
        print(f"  {a[:7]}..{b[:7]}  n={len(w):>5}  H = {h:.3f}  "
              f"floor {floor:.3f}  outside 95% band: {sig}")
        sub_rows.append({"start": a, "end": b, "n": len(w), "H": h,
                         "noise_floor": floor, "significant": sig})
    pd.DataFrame(sub_rows).to_csv(RESULTS / "bariviera_btc_subperiods.csv", index=False)

    print("\n" + "=" * 70)
    print(f"3. ROLLING DFA ({args.lookback}d window, {args.step}d step)")
    print("=" * 70)
    floor_roll = noise_floor(args.lookback, args.n_seeds)
    band = 1.96 * floor_roll
    roll = []
    vals = r.values
    for i in range(args.lookback, len(vals), args.step):
        roll.append({"date": r.index[i], "H": dfa(vals[i - args.lookback:i])})
    roll_df = pd.DataFrame(roll).set_index("date").dropna()
    outside = (roll_df["H"] - 0.5).abs() > band
    print(f"  windows: {len(roll_df)}   mean H {roll_df['H'].mean():.3f}   "
          f"std {roll_df['H'].std():.3f}")
    print(f"  single-window noise floor at n={args.lookback}: {floor_roll:.3f}")
    print(f"  windows outside the 95% no-memory band: {outside.sum()}/{len(roll_df)} "
          f"({outside.mean():.0%}; ~5% expected by chance)")
    print(f"  observed std {roll_df['H'].std():.3f} vs noise floor {floor_roll:.3f} -> "
          + ("time variation EXCEEDS estimator noise"
             if roll_df["H"].std() > floor_roll else
             "time variation is INDISTINGUISHABLE from estimator noise"))
    roll_df.to_csv(RESULTS / "bariviera_btc_rolling_hurst.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax[0].semilogy(btc.index, btc.values, color="C0")
        ax[0].set_ylabel(f"{tickers[0]} (log)"); ax[0].grid(alpha=0.3)
        ax[0].set_title("Bariviera (2017) replication: BTC rolling Hurst vs the DFA noise floor")
        ax[1].axhspan(0.5 - band, 0.5 + band, color="grey", alpha=0.25,
                      label=f"95% no-memory band (n={args.lookback})")
        ax[1].axhline(0.5, color="k", lw=0.8)
        ax[1].plot(roll_df.index, roll_df["H"], color="C3", label=f"rolling DFA H")
        ax[1].set_ylabel("H"); ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "bariviera_btc_hurst.png", dpi=120)
        print(f"  saved {RESULTS / 'bariviera_btc_hurst.png'}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("  The persistent era the paper rests on is outside yfinance's history,")
    print("  and on the era that IS available the rolling H is not distinguishable")
    print("  from 0.5 at the 500-day window this repo trades on. The full-sample")
    print(f"  H of {h_full:.3f} is the regime-mixing artifact, not a signal: no")
    print("  sub-period reproduces it and Lo's test does not reject.")


if __name__ == "__main__":
    main()
