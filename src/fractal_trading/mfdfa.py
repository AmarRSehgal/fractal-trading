"""Multifractal Detrended Fluctuation Analysis (MF-DFA).

Kantelhardt et al. (2002). Generalizes DFA by taking the q-th moment of
per-window fluctuations, yielding a spectrum h(q) instead of a single H.
For q=2 reduces to classical DFA.

The multifractal width Delta h = h(q_min) - h(q_max) quantifies how much
the scaling differs between small and large fluctuations:
 - Monofractal (e.g. fBm, fGn):       Delta h ~ 0
 - Mild multifractal (calm markets):  Delta h ~ 0.1
 - Strong multifractal (crashy):      Delta h > 0.2
"""
from __future__ import annotations

import numpy as np


def mfdfa(
    returns: np.ndarray,
    scales: np.ndarray | None = None,
    q_values: np.ndarray | None = None,
    order: int = 2,
) -> dict:
    """Compute MF-DFA h(q) spectrum.

    Returns dict with keys:
      - 'q'      : array of q values
      - 'h'      : h(q) exponents
      - 'F_q_s'  : (n_q x n_s) array of log F_q(s)
      - 'scales' : array of scales used
      - 'delta_h': max h - min h (multifractal width)
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    N = len(r)
    if N < 200:
        return {"q": np.array([]), "h": np.array([]), "delta_h": np.nan}

    if scales is None:
        scales = np.unique(np.logspace(np.log10(20), np.log10(N // 4), 12).astype(int))
    scales = np.array([s for s in scales if 2 * (order + 1) < s <= N // 2])
    if q_values is None:
        q_values = np.array([-4, -3, -2, -1, 0.5, 1, 2, 3, 4], dtype=float)

    y = np.cumsum(r - np.mean(r))
    n_s = len(scales)
    n_q = len(q_values)
    F_q_s = np.full((n_q, n_s), np.nan)

    for is_, s in enumerate(scales):
        n_windows = N // s
        if n_windows < 2:
            continue
        # segments from start AND from end (standard MF-DFA)
        y_fwd = y[: n_windows * s].reshape(n_windows, s)
        y_bwd = y[-n_windows * s:].reshape(n_windows, s)
        t = np.arange(s)

        F2 = []
        for seg_arr in [y_fwd, y_bwd]:
            for w in seg_arr:
                p = np.polyfit(t, w, order)
                resid = w - np.polyval(p, t)
                F2.append(np.mean(resid ** 2))
        F2 = np.array(F2)
        F2 = F2[F2 > 0]
        if len(F2) < 4:
            continue

        for iq, q in enumerate(q_values):
            if abs(q) < 1e-8:
                F_q_s[iq, is_] = np.exp(0.5 * np.mean(np.log(F2)))
            else:
                F_q_s[iq, is_] = np.mean(F2 ** (q / 2)) ** (1 / q)

    # Fit h(q) from log F_q(s) vs log s
    h = np.full(n_q, np.nan)
    for iq in range(n_q):
        row = F_q_s[iq, :]
        valid = np.isfinite(row) & (row > 0)
        if valid.sum() < 4:
            continue
        slope, _ = np.polyfit(np.log(scales[valid]), np.log(row[valid]), 1)
        h[iq] = slope

    valid_h = np.isfinite(h)
    delta_h = (h[valid_h].max() - h[valid_h].min()) if valid_h.any() else np.nan

    return {
        "q": q_values,
        "h": h,
        "F_q_s": F_q_s,
        "scales": scales,
        "delta_h": float(delta_h),
    }


def delta_h_only(returns: np.ndarray, **kwargs) -> float:
    """Convenience scalar wrapper: just return Delta h."""
    res = mfdfa(returns, **kwargs)
    return res.get("delta_h", np.nan)
