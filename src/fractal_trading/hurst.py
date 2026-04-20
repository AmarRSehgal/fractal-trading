"""Hurst exponent estimators: DFA and Lo's modified R/S.

DFA is the workhorse estimator (more robust to trends than classical R/S).
Modified R/S is the skeptical check - use both and compare per Lo (1991).
"""
from __future__ import annotations

import numpy as np


def dfa(
    returns: np.ndarray,
    scales: np.ndarray | None = None,
    order: int = 1,
) -> float:
    """Detrended fluctuation analysis Hurst exponent.

    Parameters
    ----------
    returns : array of log returns (or any stationary increment series)
    scales  : window sizes to use. Default: log-spaced from 10 to N/4.
    order   : polynomial order for detrending within each window (1 = linear)

    Returns
    -------
    H : float, slope of log F(s) vs log s. ~0.5 for random walk,
        >0.5 for persistent (trending), <0.5 for anti-persistent (mean-rev).
    Returns np.nan if the input is too short or degenerate.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    N = len(returns)
    if N < 100:
        return np.nan

    if scales is None:
        scales = np.unique(
            np.logspace(np.log10(10), np.log10(N // 4), 20).astype(int)
        )
    scales = np.asarray([s for s in scales if 2 * (order + 1) < s <= N // 2])
    if len(scales) < 4:
        return np.nan

    y = np.cumsum(returns - np.mean(returns))

    F = np.zeros(len(scales))
    for i, s in enumerate(scales):
        n_windows = N // s
        if n_windows < 2:
            F[i] = np.nan
            continue
        y_trim = y[: n_windows * s].reshape(n_windows, s)
        t = np.arange(s)
        rms = np.empty(n_windows)
        for j, w in enumerate(y_trim):
            p = np.polyfit(t, w, order)
            resid = w - np.polyval(p, t)
            rms[j] = np.sqrt(np.mean(resid ** 2))
        F[i] = np.sqrt(np.mean(rms ** 2))

    valid = np.isfinite(F) & (F > 0)
    if valid.sum() < 4:
        return np.nan

    slope, _ = np.polyfit(np.log(scales[valid]), np.log(F[valid]), 1)
    return float(slope)


def _andrews_bandwidth(x: np.ndarray) -> int:
    """Andrews (1991) data-dependent bandwidth for the Newey-West long-run
    variance estimator. AR(1) plug-in version."""
    x = x - np.mean(x)
    rho = np.sum(x[1:] * x[:-1]) / np.sum(x[:-1] ** 2)
    rho = np.clip(rho, -0.97, 0.97)
    alpha = 4 * rho ** 2 / ((1 - rho ** 2) ** 2)
    q = int(np.ceil(1.1447 * (alpha * len(x)) ** (1 / 3)))
    return max(1, min(q, len(x) // 4))


def _long_run_std(x: np.ndarray, q: int) -> float:
    """Newey-West long-run standard deviation with Bartlett weights."""
    x = x - np.mean(x)
    n = len(x)
    gamma0 = np.mean(x ** 2)
    var = gamma0
    for j in range(1, q + 1):
        w = 1.0 - j / (q + 1)
        gamma_j = np.sum(x[j:] * x[:-j]) / n
        var += 2 * w * gamma_j
    return float(np.sqrt(max(var, 1e-16)))


def modified_rs(returns: np.ndarray, q: int | None = None) -> dict:
    """Lo (1991) modified R/S statistic.

    Null: no long-range dependence. Under the null, Q_n / sqrt(n) is
    asymptotically a Brownian bridge range with 95% critical interval
    roughly [0.809, 1.862]. Values above 1.862 indicate persistence.

    Parameters
    ----------
    returns : array of returns
    q : Newey-West lag. If None, uses Andrews data-dependent rule.

    Returns
    -------
    dict with keys:
      - 'Q'        : the V_n = Q_n / sqrt(n) statistic (compare to [0.809, 1.862])
      - 'q'        : lag used
      - 'rejects_null_95' : True if V_n outside [0.809, 1.862]
      - 'H_implied' : rough Hurst estimate via log(Q_n) / log(n) + 0.5
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 100:
        return {"Q": np.nan, "q": np.nan, "rejects_null_95": False, "H_implied": np.nan}

    if q is None:
        q = _andrews_bandwidth(r)

    mean = np.mean(r)
    y = np.cumsum(r - mean)
    R = y.max() - y.min()
    S = _long_run_std(r, q)
    Qn = R / S
    V = Qn / np.sqrt(n)

    rejects = V > 1.862 or V < 0.809
    H_implied = np.log(Qn) / np.log(n) if Qn > 0 else np.nan
    return {
        "Q": float(V),
        "q": int(q),
        "rejects_null_95": bool(rejects),
        "H_implied": float(H_implied),
    }
