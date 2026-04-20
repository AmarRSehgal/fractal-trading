# IDEAS: fractal and Fourier methods for trading

Concrete analysis and strategy ideas with formulas, pseudocode, data needs, and
evaluation plans. Each idea is ranked by **payoff per effort** based on
literature and likelihood of surviving out-of-sample.

See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for the source papers each idea
draws from and what I could / couldn't fetch.

---

## 0. Ground rules (anti-patterns to avoid)

Before anything else, a small list of things that *look* like fractal / Fourier
trading ideas but are not:

1. **Elliott Wave** - narrative pattern-matching, not math. Not fractal in any
   falsifiable sense.
2. **Bill Williams "fractal" indicator** - a 5-bar pivot high/low. Branding, not
   fractal analysis.
3. **FFT on raw price to find dominant cycles** - price is non-stationary and
   non-periodic; peaks you find are almost always windowing artifacts or
   spurious. Every serious critique (Lo 1991 style) has buried this.
4. **Symmetric low-pass filtering of price for signals** - pure lookahead bias.
   The filter uses future values. If you see backtested charts with smoothed
   price signals, assume lookahead until proven otherwise.
5. **Cherry-picking window sizes for Hurst until you get H > 0.5** - Hurst
   estimates are noisy, especially on < 500 samples. Always report a confidence
   interval from a surrogate (shuffled) distribution.

The null hypothesis in every experiment below is `H = 0.5` (random walk) or
"no seasonal peak above noise." You should *hope* to reject it, not assume it.

---

## Part A - Analysis ideas (estimators and diagnostics)

### A1. Rolling Hurst exponent via DFA (flagship)

**Source:** Bariviera (2017); Di Matteo et al. (2005); Lo (1991, critique).

**Idea.** DFA (detrended fluctuation analysis) estimates the Hurst exponent on
non-stationary signals better than classical R/S. Apply on a rolling window of
log returns to classify market regimes:
- `H > 0.55` - persistent / trending
- `H ~ 0.5` - efficient / random walk
- `H < 0.45` - anti-persistent / mean-reverting

**Algorithm (DFA).** Given log returns `r_t`, `t = 1..N`:

1. Integrate: `y_t = sum_{i=1..t} (r_i - mean(r))`.
2. For each window size `s` in a logarithmic grid:
   a. Split `y` into non-overlapping windows of length `s`.
   b. In each window, fit a polynomial trend (order 1 or 2) and compute the
      RMS of the residuals.
   c. Average across windows to get `F(s)`.
3. Plot `log F(s)` vs `log s`. Slope is the Hurst exponent `H`.

**Pseudocode:**

```python
def dfa(returns, scales, order=1):
    y = np.cumsum(returns - returns.mean())
    F = []
    for s in scales:
        n_windows = len(y) // s
        y_trim = y[:n_windows * s].reshape(n_windows, s)
        t = np.arange(s)
        rms_per_window = []
        for w in y_trim:
            p = np.polyfit(t, w, order)
            resid = w - np.polyval(p, t)
            rms_per_window.append(np.sqrt(np.mean(resid ** 2)))
        F.append(np.mean(rms_per_window))
    slope, _ = np.polyfit(np.log(scales), np.log(F), 1)
    return slope  # Hurst exponent estimate
```

**Data.** Daily or hourly log returns, 500-2000 samples per rolling window.
Use `nolds.dfa` as a reference implementation to sanity-check your own.

**Evaluation.**
- Run on shuffled returns (destroys serial correlation). Should yield
  `H ~ 0.5`. If not, your estimator is biased.
- Bariviera (2017) found BTC Hurst in 2011-13 was ~0.55-0.60 but drifted toward
  0.50 by 2014-17 as the market matured. Try to reproduce.
- Lo (1991) showed classical R/S inflates H when short-range autocorrelation is
  present. Compute both DFA and **modified R/S** and compare.

**Payoff estimate.** High, if used as a regime filter (see strategy S1). Low, if
used as a direct signal - H drifts slowly, won't generate many trades.

---

### A2. Modified R/S statistic (Lo 1991)

**Source:** Lo (1991), *Long-term memory in stock market prices*.

**Idea.** Classical R/S (rescaled range) famously finds long memory in equities,
but Lo showed most of these findings disappear once you correct the denominator
for short-range autocorrelation. Implement Lo's modified R/S as the skeptical
check on A1.

**Formula.** For a series of length `n` with cumulative sum `Y_k`:

```
Q_n = (1 / sigma_n(q)) * [ max_k (Y_k - (k/n) Y_n) - min_k (Y_k - (k/n) Y_n) ]
```

where `sigma_n(q)` is the **Newey-West** style long-run variance estimator:

```
sigma_n(q)^2 = gamma_0 + 2 * sum_{j=1..q} w_j(q) * gamma_j
gamma_j       = sample autocovariance at lag j
w_j(q)        = 1 - j / (q + 1)    (Bartlett weights)
```

Choose `q` via Andrews' data-dependent rule or set `q = floor(4 * (n/100)^(2/9))`.

**Evaluation.** If classical R/S gives H > 0.5 but Lo's modified statistic is
inside its null band, the "long memory" is actually short-range autocorrelation.
This is the single most important sanity check in fractal market analysis.

---

### A3. Fractionally differentiated features (López de Prado, AFML ch. 5)

**Source:** López de Prado (2018), AFML ch. 5; Hosking (1981).

**Idea.** Integer-differenced returns are stationary but erase memory. Raw
prices have memory but are non-stationary. **Fractional differencing** with
`d` in `(0, 1)` lets you keep memory *and* achieve stationarity (ADF test
passes). Use as a feature for any ML model on price series.

**Formula.** The fractional difference operator using binomial expansion:

```
(1 - L)^d x_t = sum_{k=0..inf} (-1)^k C(d, k) x_{t-k}

C(d, k) = d * (d - 1) * ... * (d - k + 1) / k!
```

In practice, use the **fixed-width window (FFD)** variant: truncate weights when
`|w_k| < tau` (e.g. `tau = 1e-5`). This avoids the expanding-window lookahead
problem and gives stationary drift-free series.

**Pseudocode:**

```python
def frac_diff_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)

def frac_diff_ffd(series, d, tau=1e-5):
    w = frac_diff_weights(d, len(series))
    # truncate where weights fall below tau
    cutoff = np.argmax(np.abs(w) < tau) if (np.abs(w) < tau).any() else len(w)
    w = w[:cutoff]
    out = np.convolve(series, w, mode='valid')
    return out
```

**How to pick `d`.** Sweep `d` from 0 to 1 in 0.05 steps, compute ADF statistic
at each `d`, and pick the smallest `d` whose ADF p-value < 0.05. Often this is
`d ~ 0.3-0.5` for prices - you get stationarity without erasing all the drift.

**Evaluation.**
- Compare predictive power of `frac_diff(price, d)` vs `log_return` vs raw
  price in a simple downstream regression (predicting next-day return from
  lagged feature).
- Check cross-correlation: fractionally-differenced series should correlate
  highly (`> 0.8`) with the original, unlike log returns which correlate near 0.

**Payoff estimate.** High. This is the one idea in the literature that is
clearly adopted by professional quant shops as a feature engineering tool.

---

### A4. Multifractal DFA (MF-DFA) for volatility regime

**Source:** Bariviera et al. on 84 cryptocurrencies
(arXiv:2003.09720); Kantelhardt et al. (2002) original method.

**Idea.** A single Hurst exponent assumes scaling is uniform. Real markets show
different scaling for small vs large fluctuations - this is multifractality.
The MF-DFA spectrum `h(q)` gives `H` at different moment orders `q`; the width
`delta h = h(q_min) - h(q_max)` quantifies multifractality.

Wide `delta h` - heavy-tailed, complex, non-stationary regime.
Narrow `delta h` - closer to monofractal / fractional Brownian motion.

**Algorithm.** Same as DFA, but step 2c becomes:

```
F_q(s) = ( mean across windows of F^2(s)^(q/2) )^(1/q)
```

for a range of `q` values, typically `q in [-5, +5] \ {0}`. The scaling
exponent `h(q)` is the slope of `log F_q(s) vs log s` for each `q`.

**Use.**
- Flag regimes where `delta h > threshold` as "complex" and avoid strategies
  that assume Gaussian noise.
- Compare pairs: if pair A has `delta h = 0.2` and pair B has `delta h = 0.5`,
  B is a worse candidate for simple moving-average strategies.

**Important caveat from the 2020 paper:** shuffling a time series *reduces* but
does not *eliminate* multifractality. That means fat tails alone produce
apparent multifractality - it is not pure evidence of long-range dependence.
Always compare `delta h(original)` vs `delta h(shuffled)`.

**Evaluation.** Reproduce the Bariviera finding that cryptocurrencies show
*heterogeneous* multifractal profiles - some are monofractal fractional
Gaussian noise, others genuinely multifractal. Pick 10 liquid crypto pairs and
plot their `h(q)` spectra side by side.

---

### A5. Spectral analysis of *stationary* features

**Source:** Classical signal processing; any textbook on Welch's method.

**Idea.** Do NOT run FFT on price. DO run FFT on stationary derived features:
- Absolute returns `|r_t|` (volatility proxy)
- Trading volume
- Bid-ask spread
- Funding rate on perpetual futures (often pseudo-periodic with the funding
  interval)

Welch's method (windowed periodogram averaging) gives a clean PSD for these.
Real peaks you should find:
- **24-hour peak** in volume/volatility - the intraday U-shape
- **168-hour peak** in volume - day-of-week effect (weaker in crypto, strong
  in equities/FX)
- **Funding cycle peak** (8-hour on most perps) - mostly from the funding
  itself leaking into price

**Pseudocode:**

```python
from scipy.signal import welch
freqs, psd = welch(abs_returns, fs=sampling_per_day, nperseg=2048)
# plot log(psd) vs freqs; look for peaks above the 1/f baseline
```

**Payoff estimate.** Medium. The findings themselves are known, but quantifying
them on your data gives you honest calibration for execution (e.g. concentrate
passive posting during high-volume windows).

---

### A6. Wavelet decomposition for multi-scale features

**Source:** Gençay, Selçuk, Whitcher (2001); Ramsey (1999).

**Idea.** Wavelets are localized in time *and* frequency (FFT is only frequency).
Use **Maximum Overlap Discrete Wavelet Transform (MODWT)** on returns to
decompose into scale-specific detail coefficients:
- `d1` - 2-4 bar fluctuations (microstructure noise)
- `d2` - 4-8 bar (short-term)
- ...
- `dK` - longest scale retained

**Uses.**
- Feature engineering: wavelet coefficients at each scale as inputs to a model.
- Denoising: zero out `d1`/`d2` detail coefficients if you believe they're
  microstructure noise, reconstruct a cleaner series. **Only use causal
  wavelets** (e.g. Haar with padding, or boundary-corrected MODWT) to avoid
  lookahead.
- Cross-scale coherence between two assets (wavelet coherence) - tells you on
  which timescales assets co-move.

**Library.** `pywavelets` (`pywt.swt` for stationary wavelet transform / MODWT).

**Payoff estimate.** Medium-low. More compelling as a research tool than a
production signal. Real quant shops rarely deploy wavelet-filtered signals
directly; they use wavelet coefficients as ML features.

---

## Part B - Fourier option pricing (tangential but rigorous)

### B1. Carr-Madan FFT pricer for European options

**Source:** Carr & Madan (1999) (cited from prior knowledge; PDF fetch returned
binary-only).

**Idea.** For any model whose characteristic function of log spot `X_T =
log(S_T)` is known in closed form (Black-Scholes, Heston, Variance Gamma,
CGMY), you can price a whole strip of European calls in O(N log N) via a
single FFT.

**Setup.** Let `phi_T(u) = E[exp(i u X_T)]` be the characteristic function of
log-spot. The damped call `c_T(k) = exp(alpha k) C_T(k)`, where `k = log(K)`,
`alpha > 0` ensures L2 integrability. Its Fourier transform is:

```
psi_T(v) = exp(-r T) * phi_T(v - (alpha + 1) i)
          / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v)
```

Then:

```
C_T(k) = (exp(-alpha k) / pi) * integral_{0..inf} Re[ exp(-i v k) psi_T(v) ] dv
```

Discretize `v_j = eta * j` and sample `k_u = -b + lambda * u` with
`lambda * eta = 2 pi / N`. The integral becomes a discrete FFT.

**Choice of `alpha`.** Carr-Madan recommend `alpha ~ 1.5`. Deep OTM options
need larger `alpha`; too large introduces numerical error. Sensible default:
`alpha = 1.5`, `eta = 0.25`, `N = 4096`.

**Pseudocode:**

```python
def carr_madan(phi_T, r, T, S0, alpha=1.5, N=4096, eta=0.25):
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2
    v = np.arange(N) * eta
    u = v - (alpha + 1) * 1j
    psi = np.exp(-r * T) * phi_T(u) / (alpha ** 2 + alpha - v ** 2
                                        + 1j * (2 * alpha + 1) * v)
    # Simpson weights for accuracy
    simpson = (3 + (-1) ** np.arange(N) - np.concatenate(([1], np.zeros(N-1)))) / 3
    x = np.exp(1j * b * v) * psi * eta * simpson
    y = np.fft.fft(x)
    k = -b + lam * np.arange(N)   # log strikes
    C = np.exp(-alpha * k) / np.pi * np.real(y)
    K = np.exp(k)                  # strikes
    return K, C
```

**Characteristic functions to implement.**
- Black-Scholes: `phi(u) = exp(i u (log S0 + (r - sigma^2/2) T) - sigma^2 u^2 T / 2)`
- Heston: closed form with complex log - use principal branch carefully.
- Variance Gamma: `phi(u) = exp(i u omega T) * (1 - i theta nu u + sigma^2 nu u^2 / 2)^(-T/nu)`

**Why bother if not trading options?** Two reasons:
1. Educational - forces you to understand characteristic functions and Fourier
   inversion rigorously.
2. If you ever want to trade Deribit BTC/ETH options, you can price Heston to
   market IV surfaces and flag mispricings.

**Evaluation.** Verify against Black-Scholes closed form to machine precision
for the BS characteristic function. Then calibrate Heston to Deribit IV
snapshots and see residuals.

---

## Part C - Trading strategy ideas

Each strategy proposes a concrete entry/exit rule, sizing, and a backtest plan.
None are production-ready; all need out-of-sample validation.

### S1. Hurst regime gate on existing strategies

**Source:** Synthesis of A1 + A4.

**Hypothesis.** Trend-following strategies work when `H > 0.55`, mean-reversion
strategies work when `H < 0.45`, neither works near `H = 0.5`.

**Setup.**
- Universe: 5-10 liquid crypto pairs or equity ETFs.
- Rolling window: 500 bars (daily) or 1000 bars (hourly).
- Signal: compute DFA Hurst `H_t` at each step.

**Rule.**
- If `H_t > 0.55`: run a simple trend strategy (e.g., 20/50 SMA crossover).
- If `H_t < 0.45`: run a simple mean-reversion (e.g., z-score of 20-bar return,
  short top decile, long bottom decile).
- Else: flat.

**Backtest plan.**
1. Split data 70/30 in-sample / out-of-sample by date.
2. Tune thresholds (0.55 / 0.45) only on in-sample.
3. Report OOS Sharpe, max drawdown, hit rate for each regime separately.
4. **Key sanity check**: do the *same rule* with randomized Hurst labels.
   If OOS Sharpe is similar, your Hurst gate is doing nothing.

**Risk.** Hurst is slow-moving. Regime transitions are rare, so your effective
sample size for validating the gate is small. Expect very wide confidence
intervals.

---

### S2. Mean-reverting pair selection via spread Hurst

**Source:** Classical pair trading + A1.

**Idea.** Pair trading assumes the spread is mean-reverting. Test this directly
with Hurst on the spread, not just cointegration tests.

**Algorithm.**
1. For all pairs in a universe, compute hedge ratio beta via rolling OLS.
2. Form spread `s_t = y_t - beta * x_t`.
3. Compute `H(s)` via DFA.
4. Trade only pairs with `H < 0.40` (strongly mean-reverting) AND Johansen
   cointegration p-value < 0.05.
5. Entry: `|z-score| > 2`. Exit: `|z-score| < 0.5` or time stop 20 bars.

**Why this beats plain cointegration.** Cointegration tells you a linear
combination is stationary; Hurst tells you *how fast* it reverts. A pair with
H = 0.49 technically cointegrated but reverts so slowly you bleed on fees.

**Evaluation.** Backtest on 2018-2023, validate on 2024-2026. Track per-pair
Hurst evolution over time - pairs that drift toward H=0.5 are dying and should
be dropped.

---

### S3. Fractional-diff features into a baseline ML model

**Source:** A3 + standard ML.

**Idea.** Replace log returns with fractionally-differenced price as the input
feature to a simple model (logistic regression or gradient-boosted tree)
predicting next-bar direction.

**Feature set (all at the bar you're predicting from):**
- `frac_diff(log_price, d=0.4)` - current level
- Lagged versions at 1, 2, 5, 20 bars
- `frac_diff(log_volume, d=0.3)` - optional

**Target.** Sign of next-bar return (binary) or next-bar return itself
(regression).

**Model.** L2-regularized logistic regression. **Do not use deep learning yet**
- the whole point is to isolate whether fractional diff as a feature gives
lift vs. plain log returns.

**Evaluation.**
- Run two identical pipelines: one with log-return features, one with
  frac-diff features.
- Walk-forward CV (not k-fold - you'll leak).
- Compare OOS accuracy / AUC / strategy Sharpe.

**Expected.** Small but consistent lift (a few bps of edge per trade) on
trending assets. If you see huge lift, you're leaking data - re-check that
frac-diff uses only past weights (FFD, not expanding window).

---

### S4. Intraday seasonality-aware passive execution

**Source:** A5.

**Idea.** Volume and spread have clear intraday seasonality (FFT peaks at 24h
and its harmonics). Use the learned PSD to allocate passive liquidity posting
to windows with best fill probability and narrowest spreads.

**Simple version:**
1. Compute average volume and average spread per minute-of-day over 30 days.
2. Rank minutes by volume / spread.
3. Post passive orders preferentially in top-quartile minutes.

**Fancier version:** Fit a seasonal component (Fourier series with 24h and 12h
terms) to log-volume and use residuals as an "abnormal volume" feature.

**Payoff.** Execution alpha, not directional alpha. Won't show up in Sharpe
directly; will show up in reduced implementation shortfall.

---

### S5. Funding-rate seasonality trade (crypto perps)

**Source:** A5 applied to perps.

**Idea.** Perpetual funding is paid every 8h on most exchanges. The hours
before funding tick often show predictable pressure as the market positions
for or against the funding payment. Compute the Fourier spectrum of past
pre-funding returns to confirm a peak at the 8h cycle, then trade accordingly.

**Test first, then trade.** If the 8h peak in pre-funding return direction is
not statistically above the noise floor on a held-out period, drop the idea.
Known effect on BTC/ETH perps in 2021 that has decayed significantly - worth
re-measuring on 2024+ data.

---

## Part D - Evaluation discipline (applies to all)

A set of checks every experiment in this repo must pass before claiming an
edge:

1. **Surrogate test.** Shuffle returns (destroys serial structure). Any Hurst
   / spectral / fractal statistic should match its theoretical null on the
   surrogate. If not, your estimator is biased.
2. **Walk-forward, not random CV.** Never use k-fold on time series.
3. **Transaction cost flag.** Always run strategies with a realistic per-trade
   cost (10 bps for crypto retail, 2-5 bps for ETFs). Many "edges" evaporate.
4. **Hurst confidence interval.** Report not just `H` but its bootstrap 95% CI.
   On 500 samples, CI width is often ~0.1 - a point estimate of 0.55 with CI
   [0.49, 0.61] is *not* evidence of long memory.
5. **Pre-register.** Before running a backtest, write down what result would
   make you believe the edge is real. This prevents "finding" a narrative for
   whatever you happen to see.

---

## Roadmap

Ordered by priority. Start top-down, don't skip.

- [ ] Implement `dfa()` and `modified_rs()` in `src/fractal_trading/hurst.py`.
- [ ] Validate both on synthetic fractional Brownian motion (known H).
- [ ] Implement `frac_diff_ffd()` and ADF-based `d` picker in `fracdiff.py`.
- [ ] Notebook: reproduce Bariviera-style rolling Hurst on BTC 2013-2026 daily.
- [ ] Notebook: MF-DFA spectrum on 10 crypto pairs, ranked.
- [ ] Notebook: Welch PSD on volume and |returns| for BTC-USDT, ETH-USDT.
- [ ] Backtest S1 (Hurst regime gate) on a small universe.
- [ ] Backtest S2 (mean-reverting pair via spread Hurst).
- [ ] Carr-Madan pricer with Black-Scholes, Heston, VG characteristic functions.
   (Educational; not in main trading path.)

Do not build a framework, a config system, or a backtester abstraction until
you have three concrete experiments running. Resist the urge to over-engineer.
