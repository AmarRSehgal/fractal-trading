"""MF-DFA on VIX: does multifractal width predict forward risk moves?

Thesis: VIX log returns may exhibit time-varying multifractality. When the
spectrum width Delta h is wide, vol dynamics are "complex" (fat-tailed,
regime-prone); when narrow, vol is calm and more Gaussian-like. Hypothesis
that wide Delta h predicts future SPY drawdowns or elevated realized vol
is a regime-signal story, not a directional one.

Outputs:
  - Rolling Delta h time series on VIX
  - Correlations with forward 21-day VIX change, SPY return, realized vol
  - Simple "high Delta h -> avoid equity" gating rule vs buy-and-hold

Usage:
    python3 scripts/10_vix_mfdfa_regime.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fractal_trading.backtest import paired_sharpe_diff_ci, sharpe_ci
from fractal_trading.data import load_prices
from fractal_trading.mfdfa import mfdfa
from fractal_trading.hurst import dfa


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def rolling_mfdfa(series: np.ndarray, dates, lookback: int = 500, step: int = 21):
    """Rolling MF-DFA; returns (date, delta_h, h_q2) tuples."""
    out = []
    for end in range(lookback, len(series), step):
        window = series[end - lookback:end]
        res = mfdfa(window)
        if np.isfinite(res["delta_h"]):
            q = res["q"]; h = res["h"]
            # h at q=2 is the classical Hurst
            h2 = float(h[np.argmin(np.abs(q - 2.0))]) if len(h) else np.nan
            out.append({
                "date": dates[end - 1],
                "delta_h": res["delta_h"],
                "h_at_q2": h2,
            })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None, help="pin for reproducibility")
    parser.add_argument("--min_warmup", type=int, default=40,
                        help="rebalances required before the causal gate trades")
    args = parser.parse_args()

    print("Loading VIX, SPY...")
    df_vix = load_prices(["^VIX"], start=args.start, end=args.end)
    df_spy = load_prices(["SPY"], start=args.start, end=args.end)
    # flatten
    vix = df_vix.iloc[:, 0].dropna()
    spy = df_spy.iloc[:, 0].dropna()
    common = vix.index.intersection(spy.index)
    vix = vix.loc[common]
    spy = spy.loc[common]
    print(f"  aligned: {len(common)} days, {common.min()} to {common.max()}")

    vix_rets = np.log(vix).diff().dropna().values
    vix_dates = np.log(vix).diff().dropna().index

    print("\nFull-sample MF-DFA on VIX log returns...")
    t0 = time.time()
    res_full = mfdfa(vix_rets)
    print(f"  {time.time() - t0:.1f}s")
    print(f"  q values: {res_full['q']}")
    print(f"  h(q):     {np.round(res_full['h'], 3)}")
    print(f"  Delta h:  {res_full['delta_h']:.3f}")
    print(f"  Classical H (q=2): {res_full['h'][np.argmin(np.abs(res_full['q'] - 2.0))]:.3f}")

    # Rolling MF-DFA
    print("\nRolling MF-DFA (lookback=500d, step=21d)...")
    t0 = time.time()
    roll = rolling_mfdfa(vix_rets, vix_dates, lookback=500, step=21)
    print(f"  {time.time() - t0:.1f}s; {len(roll)} rolling points")

    # Build forward-return panel aligned to roll dates
    roll["date"] = pd.to_datetime(roll["date"])
    roll.set_index("date", inplace=True)

    spy_logp = np.log(spy)
    vix_logp = np.log(vix)

    # forward 21-day log returns
    spy_fwd = spy_logp.shift(-21) - spy_logp
    vix_fwd = vix_logp.shift(-21) - vix_logp

    # realized 21d forward vol of SPY
    spy_rets = np.log(spy).diff()
    spy_fwd_vol = spy_rets.rolling(21).std().shift(-21) * np.sqrt(252)

    # align
    df = roll.join(pd.DataFrame({
        "spy_fwd_ret": spy_fwd,
        "vix_fwd_ret": vix_fwd,
        "spy_fwd_vol": spy_fwd_vol,
        "vix_level": vix,
    }), how="left").dropna()

    print("\n=== Correlations of Delta h with forward 21-day outcomes ===")
    for col in ["spy_fwd_ret", "vix_fwd_ret", "spy_fwd_vol", "vix_level"]:
        r = df["delta_h"].corr(df[col])
        print(f"  delta_h vs {col:<15s}  rho = {r:+.3f}  (n={len(df)})")
    print("\n=== Same correlations for classical H (q=2) ===")
    for col in ["spy_fwd_ret", "vix_fwd_ret", "spy_fwd_vol", "vix_level"]:
        r = df["h_at_q2"].corr(df[col])
        print(f"  h_q=2  vs {col:<15s}  rho = {r:+.3f}")

    # === Regime gate ===
    #
    # LOOKAHEAD WARNING: the original version of this script thresholded
    # delta_h on its FULL-SAMPLE median, so the 2008 gate depended on 2026
    # data. That is the leak that made this look like the best signal in the
    # repo. Both versions are reported below; only `causal` is tradeable.
    #
    # Observations are non-overlapping by construction: rolling_mfdfa steps
    # 21 trading days and the forward return spans exactly 21 trading days,
    # so consecutive rows abut rather than overlap. An i.i.d. bootstrap is
    # therefore admissible here; we still report a block bootstrap as a
    # dependence-robust check.
    df["bh_ret"] = df["spy_fwd_ret"]

    # (a) in-sample / leaky: full-sample median
    med_full = df["delta_h"].median()
    df["gate_leaky"] = np.where(df["delta_h"] <= med_full, df["bh_ret"], 0.0)

    # (b) causal: expanding median using only rebalances already observed
    expanding_med = df["delta_h"].expanding().median().shift(1)
    in_mkt = df["delta_h"] <= expanding_med
    in_mkt.iloc[: args.min_warmup] = True   # no threshold yet -> hold, like BH
    df["expanding_med"] = expanding_med
    df["gate_causal"] = np.where(in_mkt, df["bh_ret"], 0.0)

    PPY = 252 / 21          # non-overlapping 21-day periods per year
    def _line(name, r):
        ci = sharpe_ci(r.values, periods_per_year=PPY, n_boot=5000, seed=1)
        return (f"  {name:<22s} ann_ret {r.mean() * PPY:+.3f}  "
                f"ann_vol {r.std() * np.sqrt(PPY):.3f}  "
                f"Sharpe {ci['point']:.2f}  95% CI [{ci['lo']:+.2f}, {ci['hi']:+.2f}]")

    print("\n=== Regime gate: 'flat when delta_h above median' ===")
    print(_line("buy-and-hold SPY", df["bh_ret"]))
    print(_line("gate (LEAKY median)", df["gate_leaky"]))
    print(_line("gate (causal median)", df["gate_causal"]))
    print(f"  time in market: leaky {(df['gate_leaky'] != 0).mean() * 100:.0f}%  "
          f"causal {(df['gate_causal'] != 0).mean() * 100:.0f}%")

    # Paired bootstrap of the Sharpe DIFFERENCE -- this is the claim being
    # made ("the gate beats buy-and-hold"), and it must be paired because the
    # two series share the same underlying SPY returns.
    print("\n=== Paired bootstrap: Sharpe(gate) - Sharpe(buy-and-hold) ===")
    for label, col in (("leaky", "gate_leaky"), ("causal", "gate_causal")):
        for blk in (1, 4):
            d = paired_sharpe_diff_ci(
                df[col].values, df["bh_ret"].values,
                periods_per_year=PPY, n_boot=5000, seed=1, expected_block=blk,
            )
            kind = "iid  " if blk == 1 else "block"
            print(f"  {label:<7s} {kind}  diff {d['point']:+.3f}  "
                  f"95% CI [{d['lo']:+.3f}, {d['hi']:+.3f}]  "
                  f"P(diff>0) {d['p_gt_0']:.1%}  n={d['n']}")

    # Save
    df.to_csv(RESULTS / "vix_mfdfa_regime.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(df.index, df["vix_level"], color="gray")
        axes[0].set_ylabel("VIX level")
        axes[0].grid(alpha=0.3)
        axes[1].plot(df.index, df["delta_h"], color="C0", label="Delta h")
        axes[1].plot(df.index, df["h_at_q2"], color="C1", alpha=0.6, label="h(q=2)")
        axes[1].plot(df.index, df["expanding_med"], color="red", linestyle="--", alpha=0.6, label="expanding median")
        axes[1].set_ylabel("MF-DFA exponents")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(df.index, df["spy_fwd_vol"], color="purple")
        axes[2].set_ylabel("SPY 21d fwd realized vol (ann)")
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "vix_mfdfa_regime.png", dpi=120)
        print(f"Saved: {RESULTS / 'vix_mfdfa_regime.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
