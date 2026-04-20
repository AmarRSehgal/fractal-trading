"""Validate fractional differencing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fractal_trading.fracdiff import frac_diff_weights, frac_diff_ffd, find_min_d
from fractal_trading.synthetic import random_walk


def test_weights_d_zero_is_identity():
    """d=0 should give only w_0=1 (series unchanged)."""
    w = frac_diff_weights(0.0)
    assert len(w) == 1 and abs(w[0] - 1.0) < 1e-12


def test_weights_d_one_matches_first_difference():
    """d=1 gives w = [1, -1, 0, 0, ...]; truncated at second term."""
    w = frac_diff_weights(1.0)
    # w_0=1, w_1=-1*(1-0)/1 = -1, w_2 = -(-1)*(1-1)/2 = 0 -> truncated
    assert len(w) >= 2
    assert abs(w[0] - 1.0) < 1e-12
    assert abs(w[1] - (-1.0)) < 1e-12


def test_weights_decay_for_fractional_d():
    """For 0 < d < 1, weights should decay (in magnitude) eventually."""
    w = frac_diff_weights(0.4, tau=1e-5)
    assert len(w) > 5
    assert abs(w[-1]) < abs(w[0])


def test_ffd_makes_random_walk_stationary():
    """Random walk (non-stationary) should become stationary for some d > 0."""
    rw = random_walk(3000, seed=42)
    d_star = find_min_d(rw)
    assert not np.isnan(d_star), "Failed to find stationarity d"
    assert 0.0 < d_star <= 1.0, f"Unexpected d_star={d_star}"


def test_ffd_preserves_correlation_with_original():
    """FFD output should correlate highly with original series - unlike
    integer differencing which usually has ~0 correlation."""
    rw = random_walk(3000, seed=7)
    y_fd = frac_diff_ffd(rw, d=0.4)
    y_fd_valid = y_fd[~np.isnan(y_fd)]
    rw_aligned = rw[-len(y_fd_valid):]
    corr_ffd = np.corrcoef(y_fd_valid, rw_aligned)[0, 1]

    y_int = np.diff(rw)
    rw_int_aligned = rw[1:]
    corr_int = np.corrcoef(y_int, rw_int_aligned)[0, 1]

    assert abs(corr_ffd) > abs(corr_int), \
        f"FFD should correlate more with raw series: ffd={corr_ffd}, int={corr_int}"


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
