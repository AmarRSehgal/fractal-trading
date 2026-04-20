"""Fractional differentiation (López de Prado AFML ch. 5, FFD variant).

FFD = fixed-width window. Truncate weights when they fall below tau. Gives
a drift-free stationary output that retains long-term memory from the raw
series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def frac_diff_weights(d: float, tau: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """Binomial-expansion weights for fractional differencing.

    w_k = -w_{k-1} * (d - k + 1) / k, w_0 = 1. Truncated when |w_k| < tau.
    """
    w = [1.0]
    for k in range(1, max_size):
        next_w = -w[-1] * (d - k + 1) / k
        if abs(next_w) < tau:
            break
        w.append(next_w)
    return np.asarray(w)


def frac_diff_ffd(series: np.ndarray | pd.Series, d: float, tau: float = 1e-5) -> np.ndarray:
    """Fractionally differenced series via fixed-width window.

    Output has NaN in the first len(weights)-1 entries. Weights apply to
    current and past values only - no lookahead.
    """
    x = np.asarray(series, dtype=float)
    w = frac_diff_weights(d, tau=tau, max_size=len(x))
    K = len(w)
    if K >= len(x):
        return np.full(len(x), np.nan)
    # convolve(x, w, 'valid') == sum_k x[t-k] * w[k] for t >= K-1
    y = np.convolve(x, w, mode="valid")
    out = np.full(len(x), np.nan)
    out[K - 1:] = y
    return out


def find_min_d(
    series: np.ndarray | pd.Series,
    d_grid: np.ndarray | None = None,
    p_threshold: float = 0.05,
    tau: float = 1e-5,
) -> float:
    """Smallest d in d_grid for which the FFD series passes ADF stationarity.

    Returns np.nan if no d in the grid achieves stationarity.
    """
    if d_grid is None:
        d_grid = np.arange(0.0, 1.0 + 1e-9, 0.05)

    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 200:
        return np.nan

    for d in d_grid:
        y = frac_diff_ffd(x, d, tau=tau)
        y = y[np.isfinite(y)]
        if len(y) < 100:
            continue
        try:
            pval = adfuller(y, autolag="AIC")[1]
        except Exception:
            continue
        if pval < p_threshold:
            return float(d)
    return np.nan
