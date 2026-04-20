"""Print validation report for DFA and modified R/S on synthetic series.

Run: python3 scripts/01_validate_estimators.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fractal_trading.hurst import dfa, modified_rs
from fractal_trading.synthetic import white_noise, ar1, fbm_increments


def summarize(label, estimates):
    a = np.array(estimates)
    return f"{label:<25s} mean={a.mean():.3f}  std={a.std():.3f}  min={a.min():.3f}  max={a.max():.3f}"


def main():
    print("DFA Hurst estimator on synthetic series (n=4000, 20 seeds)")
    print("-" * 72)

    # Known-true series
    for true_H in [0.30, 0.50, 0.60, 0.70, 0.80]:
        ests = [dfa(fbm_increments(4000, H=true_H, seed=s)) for s in range(20)]
        print(summarize(f"fBm H={true_H:.2f}", ests))

    print()
    print("Modified R/S test (Lo 1991) on synthetic series")
    print("-" * 72)

    wn_rej = sum(modified_rs(white_noise(2000, seed=s))["rejects_null_95"] for s in range(50))
    print(f"White noise (H=0.5):      rejections = {wn_rej}/50  (expect ~2-3 at 5% level)")

    ar1_rej = sum(modified_rs(ar1(2000, phi=0.5, seed=s))["rejects_null_95"] for s in range(50))
    print(f"AR(1) phi=0.5 (SHORT mem): rejections = {ar1_rej}/50  (Lo's point: modified R/S doesn't spuriously reject)")

    # Compare naive vs modified on AR(1) to show the correction works
    naive_H = np.mean([modified_rs(ar1(2000, phi=0.5, seed=s), q=0)["H_implied"] for s in range(20)])
    mod_H = np.mean([modified_rs(ar1(2000, phi=0.5, seed=s))["H_implied"] for s in range(20)])
    print(f"\nAR(1) implied H:  naive (q=0)={naive_H:.3f}  modified (Andrews q)={mod_H:.3f}")
    print("(Naive inflates H for AR(1); modified corrects it - Lo 1991's main point.)")


if __name__ == "__main__":
    main()
