"""Validate MF-DFA on synthetic series."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fractal_trading.mfdfa import mfdfa
from fractal_trading.synthetic import fbm_increments, white_noise


def test_mfdfa_white_noise_is_monofractal():
    """WN is monofractal: h(q) should be ~constant = 0.5."""
    x = white_noise(4000, seed=0)
    res = mfdfa(x)
    h = res["h"]
    h = h[np.isfinite(h)]
    assert len(h) >= 5, f"too few h values: {len(h)}"
    assert 0.4 < h.mean() < 0.6, f"WN h mean {h.mean()}"
    # narrow multifractal width
    assert res["delta_h"] < 0.25, f"WN delta_h = {res['delta_h']} should be small"


def test_mfdfa_fbm_075_is_monofractal_persistent():
    """fBm with H=0.75 is also monofractal but at H=0.75."""
    x = fbm_increments(4000, H=0.75, seed=0)
    res = mfdfa(x)
    h = res["h"]
    h = h[np.isfinite(h)]
    assert 0.65 < h.mean() < 0.85, f"fBm(0.75) h mean {h.mean()}"
    assert res["delta_h"] < 0.30


def test_mfdfa_q_monotonicity_on_skewed_series():
    """For a heavy-tailed monofractal series (t-distributed returns), h(q)
    should show a MILD decreasing trend with q (small departures from
    monofractal). Mostly validates that the q iteration produces sensible
    decreasing values rather than total noise."""
    rng = np.random.default_rng(0)
    # heavy-tailed returns
    x = rng.standard_t(df=3, size=4000)
    res = mfdfa(x)
    q = res["q"]; h = res["h"]
    valid = np.isfinite(h)
    # At least some ordering: typically h decreases as q increases for
    # heavy-tailed processes. Assert not-strictly-monotonically-increasing.
    # (Weak check - primary validation is the WN and fBm tests above.)
    assert valid.sum() >= 5
    diffs = np.diff(h[valid])
    # Not all differences should be positive (pure increasing would be bad)
    assert (diffs > 0).sum() < len(diffs), f"h(q) improbably monotone up: {h[valid]}"


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
