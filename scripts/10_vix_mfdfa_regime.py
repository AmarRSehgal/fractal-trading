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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

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
    print("Loading VIX, SPY, UVXY/VXX...")
    df_vix = load_prices(["^VIX"], start="2005-01-01")
    df_spy = load_prices(["SPY"], start="2005-01-01")
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

    # Naive trading rule test: high Delta h -> stay in cash, low -> long SPY
    med = df["delta_h"].median()
    df["regime"] = (df["delta_h"] > med).astype(int)  # 1 if complex
    # Holding-period return of "stay in SPY only when delta_h <= median"
    df["strategy_ret"] = np.where(df["regime"] == 0, df["spy_fwd_ret"], 0.0)
    bh_ret = df["spy_fwd_ret"].mean() * (252 / 21)
    bh_vol = df["spy_fwd_ret"].std() * np.sqrt(252 / 21)
    st_ret = df["strategy_ret"].mean() * (252 / 21)
    st_vol = df["strategy_ret"].std() * np.sqrt(252 / 21)
    print(f"\n=== Naive regime gate ('flat when delta_h above median') ===")
    print(f"  Buy-and-hold SPY: ann_ret {bh_ret:+.3f}  ann_vol {bh_vol:.3f}  Sharpe {bh_ret / bh_vol:.2f}")
    print(f"  Regime-gated:     ann_ret {st_ret:+.3f}  ann_vol {st_vol:.3f}  Sharpe {st_ret / st_vol:.2f}")
    print(f"  Time-in-market: {(df['regime'] == 0).mean() * 100:.0f}%")

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
        axes[1].axhline(df["delta_h"].median(), color="red", linestyle="--", alpha=0.5, label="median")
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
