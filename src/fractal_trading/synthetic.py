"""Synthetic series generators for estimator validation."""
from __future__ import annotations

import numpy as np


def white_noise(n: int, seed: int = 0) -> np.ndarray:
    """White noise returns. True Hurst = 0.5."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def ar1(n: int, phi: float = 0.5, seed: int = 0) -> np.ndarray:
    """AR(1) returns. Short-range dependence only; true long-memory H = 0.5
    but classical R/S will *incorrectly* give H > 0.5. Useful to demonstrate
    Lo's critique."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def fbm_increments(n: int, H: float, seed: int = 0) -> np.ndarray:
    """Fractional Brownian motion increments (fGn) via Davies-Harte.

    Returns a length-n sample of fractional Gaussian noise with given H.
    Verifies by circulant embedding of the autocovariance.
    """
    rng = np.random.default_rng(seed)
    N = 1 << int(np.ceil(np.log2(2 * n)))
    k = np.arange(N // 2 + 1)

    def gamma(h: int) -> float:
        return 0.5 * (abs(h - 1) ** (2 * H) - 2 * abs(h) ** (2 * H) + abs(h + 1) ** (2 * H))

    r = np.array([gamma(h) for h in range(N // 2 + 1)])
    r_full = np.concatenate([r, r[-2:0:-1]])
    lam = np.fft.fft(r_full).real
    if (lam < -1e-10).any():
        raise RuntimeError("Non-PSD circulant; increase N or try smaller H range.")
    lam = np.clip(lam, 0, None)

    z = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y = np.fft.fft(np.sqrt(lam) * z) / np.sqrt(N)
    sample = y[:n].real
    return sample


def random_walk(n: int, seed: int = 0) -> np.ndarray:
    """Cumulative white noise. Use when you need a price-like non-stationary
    series (for frac-diff tests). True H on increments = 0.5."""
    return np.cumsum(white_noise(n, seed=seed))
