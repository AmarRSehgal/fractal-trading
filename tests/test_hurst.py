"""Validate DFA and modified R/S on synthetic series with known properties."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fractal_trading.hurst import dfa, modified_rs
from fractal_trading.synthetic import white_noise, ar1, fbm_increments


def test_dfa_white_noise():
    """Random walk increments: H should be ~0.5."""
    estimates = [dfa(white_noise(5000, seed=s)) for s in range(10)]
    mean_H = np.mean(estimates)
    assert 0.45 < mean_H < 0.55, f"DFA on WN gave H={mean_H}, expected ~0.5"


def test_dfa_fbm_persistent():
    """fBm increments with H=0.75: DFA should recover H in (0.65, 0.85)."""
    estimates = [dfa(fbm_increments(4000, H=0.75, seed=s)) for s in range(10)]
    mean_H = np.mean(estimates)
    assert 0.65 < mean_H < 0.85, f"DFA on fBm(0.75) gave H={mean_H}"


def test_dfa_fbm_antipersistent():
    """fBm increments with H=0.3: DFA should recover H in (0.20, 0.40)."""
    estimates = [dfa(fbm_increments(4000, H=0.3, seed=s)) for s in range(10)]
    mean_H = np.mean(estimates)
    assert 0.20 < mean_H < 0.40, f"DFA on fBm(0.3) gave H={mean_H}"


def test_modified_rs_white_noise():
    """WN should generally NOT reject the null."""
    results = [modified_rs(white_noise(2000, seed=s)) for s in range(20)]
    rejections = sum(r["rejects_null_95"] for r in results)
    # at alpha=0.05 we expect ~1 rejection in 20 tries from both tails
    assert rejections <= 4, f"Too many false rejections on WN: {rejections}/20"


def test_modified_rs_corrects_ar1():
    """Key finding from Lo (1991): classical R/S says AR(1) has long memory,
    but modified R/S with Newey-West does not. Here we just verify the
    modified stat's H_implied is closer to 0.5 than the raw R/S would suggest.
    """
    # strongly AR(1) series
    series = ar1(2000, phi=0.6, seed=0)
    # Naive R/S (q=0) vs modified (q=Andrews)
    naive = modified_rs(series, q=0)
    modified = modified_rs(series)
    # Naive (classical) R/S-implied H should be inflated vs modified
    assert naive["H_implied"] > modified["H_implied"], \
        f"Expected naive H > modified H; got naive={naive}, mod={modified}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
            except Exception as e:
                print(f"ERROR {name}: {e.__class__.__name__}: {e}")
