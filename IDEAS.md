# IDEAS: fractal and Fourier methods for stock trading

Concrete analysis and strategy ideas for **US equities**, grounded in the
papers covered in [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md). Each idea is
ranked by payoff-per-effort.

This file now also contains **first-pass empirical findings** from the
backtests in `scripts/` (see [`RESULTS.md`](RESULTS.md) for the raw numbers).

---

## The constraint you must internalize first

Lo (1991) — his *Econometrica* paper on long-term memory in stock market
prices — is specifically about US equities. He showed that the aggregate US
stock market has no significant long-range dependence once you correct for
short-range autocorrelation with a Newey-West style long-run variance. This
finding has held up across 30+ years of follow-up work.

**Practical consequence:** any strategy whose pitch is "run DFA on SPY and
switch between trend and mean-reversion" is near-certain to fail. Our own
reproduction of Lo's finding is embedded in the empirical note for idea #1
below — the H-distribution across S&P 100 is tight around ~0.47 with almost
no stocks rejecting the null.

The edge, if it exists, has to come from:
- **cross-sectional dispersion** (individual stocks differing from the index)
- **stationary derived features** (vol, volume, spreads), not price memory
- **execution timing** (intraday seasonality in microstructure)

---

## Ground rules (anti-patterns)

1. **Elliott Wave** - narrative pattern matching, not math.
2. **Bill Williams "fractal" indicator** - a 5-bar pivot, not fractal analysis.
3. **FFT on raw price to find "cycles"** - price is non-stationary and
   non-periodic; peaks are windowing artifacts.
4. **Symmetric low-pass filtering of price for signals** - pure lookahead bias.
5. **Hurst on SPY/QQQ as a trend/MR switch** - Lo proved this has no signal.
6. **Cherry-picking lookback windows until H > 0.5** - always report a
   bootstrap or surrogate confidence interval.
7. **Backtests on current-constituents universes** - survivorship bias.
   yfinance-sourced data has this problem; so does every retail stock data
   source. **All backtests in this repo are biased upward** until we move to
   a delisting-aware source.

Null hypothesis in every experiment: `H = 0.5`, or "no factor edge." You
should *hope* to reject it, not assume it.

---

## Part A - Analysis ideas (estimators and diagnostics)

### A1. Rolling Hurst exponent via DFA on individual stocks

**Source:** Di Matteo et al. (2005) scaling of developed vs emerging markets;
Bariviera (2017) methodology; Lo (1991) critique.

**Idea.** DFA estimates the Hurst exponent on non-stationary signals better
than classical R/S. Compute per-stock on rolling returns as a diagnostic
and factor.

**Our DFA is validated** (see `tests/test_hurst.py` and
`scripts/01_validate_estimators.py`):
- Recovers H = 0.30, 0.50, 0.60, 0.70, 0.80 on fBm with std ~0.025.

**Empirical note on US equities.** When run on S&P 100 daily log returns
(2005-2026, full sample), DFA gives:
- Mean H = 0.465, std = 0.034
- 5-95 percentile: [0.415, 0.517]
- Only 5/100 stocks reject Lo's no-long-memory null at 95% (vs ~5 expected
  by chance alone)

That is essentially a reproduction of Lo's 1991 finding. The cross-sectional
dispersion of H is tiny. Treat H as a feature with very low information
content on large-cap US stocks.

**Code:** `fractal_trading.hurst.dfa`. Implementation details:
1. Integrate returns: `y_t = sum_{i<=t} (r_i - mean)`.
2. For each window size `s`, split `y` into non-overlapping windows, fit
   polynomial (default order 1), compute per-window RMS of residuals, then
   RMS across windows -> `F(s)`.
3. Regress `log F(s)` on `log s`. Slope = Hurst.

---

### A2. Modified R/S (Lo 1991)

**Source:** Lo (1991).

**Idea.** Classical R/S inflates H for AR(1)-like series. Modified R/S uses
a Newey-West long-run variance denominator with Andrews' data-dependent
bandwidth `q`.

**Our implementation is validated:**
- Size under H0: 2/50 rejections at 5% (correct size).
- AR(1) phi=0.5: naive implied H = 0.591 (false positive); modified = 0.525
  (correctly reined in). This is Lo's headline finding reproduced.

**Use.** Run alongside DFA. If DFA says H > 0.55 but modified R/S does not
reject the null, the apparent "long memory" is short-range autocorrelation.

**Code:** `fractal_trading.hurst.modified_rs`.

---

### A3. Fractionally differenced features (López de Prado AFML ch. 5)

**Source:** López de Prado (2018).

**Idea.** Integer differencing destroys memory. Fractional differencing with
`d` in `(0, 1)` makes a series stationary while retaining long-term
correlation with the original. Use as an ML feature.

**Our implementation is validated** (`tests/test_fracdiff.py`):
- `d = 0` gives identity; `d = 1` gives first-difference.
- FFD of a random walk is stationary (ADF rejects non-stationarity at some
  `d < 1`).
- FFD correlates highly with original series; integer-differenced does not.

**Empirical note.** As a **standalone cross-sectional factor** (z-score of
FFD(log price) with d=0.4 over 252-day rolling window), frac-diff did
**worse** than 12-1 momentum on S&P 100 2005-2026:
- Momentum 12-1: Sharpe 0.17, turnover 38%/month
- FFD z-score:   Sharpe 0.01, turnover 74%/month
- Combined:      Sharpe -0.02

FFD is noisier than a log-return momentum for a single-factor sort. My prior
was "similar edge, lower turnover"; actual result was "no edge, higher
turnover." This does NOT invalidate López de Prado's original proposal -
FFD was proposed as an ML feature for models that can use levels, not as a
standalone cross-sectional score. That test (combining FFD with a tree-based
model and walk-forward CV) is still TBD.

**Code:** `fractal_trading.fracdiff.frac_diff_ffd` and `.find_min_d`.

---

### A4. Spectral analysis of stationary derived features

**Source:** Classical signal processing; Welch's method.

**Idea.** Run FFT on **stationary** features, never raw price:
- `|log returns|` (volatility proxy)
- Volume
- Bid-ask spread

Real peaks you should find in US equities:
- **Daily peak** in volume/volatility - the intraday U-shape (open and
  close auctions concentrate liquidity).
- **Weekly** (5 trading days) peak in volume.
- **Quarterly earnings** concentration.

**Usage.** Execution alpha (not directional). Time discretionary orders to
high-liquidity windows.

**Data constraint with yfinance.** yfinance 1-minute bars are limited to
the last 7 days. For a meaningful intraday study you need a paid source
(Polygon, Alpha Vantage Premium). A scoped-down alternative: 1-hour bars
for 2 years, on a small watchlist. Worth building but lower priority than
the directional strategies.

---

### A5. MF-DFA (multifractal) for volatility regime

**Source:** Kantelhardt et al. (2002) MF-DFA; Bariviera 2020 heterogeneity
applied to 84 crypto - the method transfers to stocks.

**Idea.** A single H assumes uniform scaling. MF-DFA computes `h(q)` over a
range of moment orders `q`, and the width `delta h = h(q_min) - h(q_max)`
quantifies multifractality. Wider delta h = more complex dynamics.

**Use.** Compare SPX index vs sector ETFs vs individual stocks. Sectors with
more multifractal volatility may need different options strategies
(straddles etc.) than plain-vanilla monofractal sectors.

**Status:** Not yet implemented. Tier-2 priority given the weak directional
results.

---

## Part B - Trading strategy ideas (stock-focused)

### S1. Cross-sectional Hurst sort

**Paper basis:** Di Matteo 2005 (cross-market dispersion); Bariviera 2020
(heterogeneity).

**Hypothesis.** Within a stock universe, high-H names trend and low-H names
mean-revert. Long top-quintile H, short bottom-quintile, monthly rebalance,
equal-weighted.

**Empirical result (S&P 100, 2005-2026, lookback 500d, no costs, survivorship-biased):**

| Stat            | Value   |
|-----------------|---------|
| Annual return   | 4.9%    |
| Annual vol      | 10.5%   |
| Sharpe          | 0.47    |
| Max drawdown    | -28.1%  |
| Hit rate        | 51.4%   |
| Turnover / reb  | 42.3%   |
| N monthly rebals| 212     |

**Honest assessment.** Sharpe 0.47 before costs, on a survivorship-biased
universe, is a **null or very weak positive result**. With 42% monthly
turnover and realistic costs (say 10 bps round trip), cost drag is
~0.42 * 0.002 * 12 = 1.0% annual - knocking Sharpe to ~0.35. Survivorship
correction typically costs another 1-2% of annual return. Net expected
Sharpe after both: ~0.1-0.2.

**This is probably not a trading edge on US large caps.** Might work better
on:
- Mid/small caps (Russell 2000) where dispersion in H should be wider.
- Longer lookbacks (3-year) to reduce noise.
- Longer holding periods (quarterly) to cut turnover.
- As a **gating filter** rather than sort (only trade names with H > 0.55).

**Code:** `scripts/03_hurst_sort_backtest.py`.

---

### S2. Fractional-diff feature in an ML model

**Paper basis:** López de Prado (2018) AFML ch. 5.

**Hypothesis (revised).** As a standalone cross-sectional z-score it fails
(see A3). As an **input feature to a nonlinear model** - alongside standard
factors - it may add lift by providing level information that log-returns
erase.

**Test to run (not yet built):**
- Build a panel dataset: per-stock monthly features including frac-diff at
  multiple `d` values, log-returns at multiple horizons, vol, size.
- Fit a gradient-boosted tree predicting next-month return.
- Compare OOS hit-rate, Sharpe of factor portfolio built from predictions,
  with frac-diff included vs excluded.
- Walk-forward CV, no k-fold.

**Status:** Not yet built. Priority 2.

---

### S3. Lo-filtered trend universe

**Paper basis:** Lo (1991) - use as a filter rather than a dismissal.

**Hypothesis.** Among 3000+ US stocks, the ~5-15% that actually pass Lo's
modified R/S test at 95% have genuine short-to-medium memory. Simple trend
rules on that subset should outperform the broader universe.

**Caveat reinforced by our S&P 100 diagnostic.** Only 5/100 S&P 100 names
reject the null. Extending to Russell 3000 may give us 150-450 passers but
they are likely small-caps where trading costs are much higher.

**Test to run (not yet built):**
- Monthly: compute modified R/S on 2-year return history for Russell 3000
  (yfinance workable if slow).
- Filter to passers at 95%.
- 200-day SMA trend rule on passers.
- Compare to same rule on all 3000.

**Status:** Not yet built. Priority 2.

---

### S4. Intraday seasonality-aware execution

**Paper basis:** A4.

**Hypothesis.** Liquidity concentrates at specific intraday times in US
equities (open auction, 10:00-10:30, 15:30-16:00 close auction). A learned
intraday volume/spread profile per stock should reduce execution cost 5-20
bps per trade.

**Data constraint.** yfinance 1-minute = 7 days only. Workable scope:
hourly bars for 2 years per stock.

**Status:** Not yet built. Priority 3 (execution alpha, not directional).

---

### S5. Sector-level Hurst rotation

**Paper basis:** Di Matteo 2005 (scaling differs across markets).

**Hypothesis.** GICS sectors differ in persistence: defensives (utilities,
staples) may be more persistent; cyclicals less so. Overweight momentum in
high-H sectors.

**Universe:** 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, etc.).

**Status:** Not yet built. Priority 3. Small universe so signal-to-noise
of factor will be limited.

---

### S6. Carr-Madan SPX/VIX options pricer

**Paper basis:** Carr & Madan (1999).

**Hypothesis.** Not a trading edge per se; a calibration tool. Calibrate
Heston to SPX IV surface daily via Carr-Madan FFT. Options that are 2 sigma
away from the model fit are candidates for vol-arb investigation.

**Status:** Not yet built. Priority 3 (educational; SPX option markets are
very efficient, retail edge is near zero here).

---

## Part C - Evaluation discipline

Every strategy must pass:

1. **Surrogate test.** Shuffle returns; the factor's Sharpe on the surrogate
   must be indistinguishable from zero.
2. **Walk-forward, not random k-fold.** Never k-fold time series.
3. **Transaction costs.** 10 bps round-trip for retail liquid stocks at
   minimum. Consider borrow fees for shorts (easy-to-borrow: ~0.5-2% annual;
   hard-to-borrow: much worse).
4. **Survivorship correction.** Acknowledge the bias; add a haircut (1-2%
   annual return) until you can source delisting-aware data.
5. **Bootstrap confidence interval** on Sharpe and H. On 5000 samples,
   Sharpe CI half-width is ~0.2 - a reported Sharpe of 0.4 easily covers
   zero.
6. **Pre-register success criteria.** Write down, before running: "if
   Sharpe > 0.5 and t-stat on alpha > 2, the idea is worth scaling."

---

## Roadmap (revised after second empirical pass - 2026-04-20)

Done:
- [x] DFA + modified R/S estimators, validated on synthetic fBm and AR(1).
- [x] FFD + d-picker, validated on random walk.
- [x] yfinance data loader with parquet cache.
- [x] Cross-sectional sort backtest harness **with bootstrap CI and TC model**.
- [x] Hurst distribution diagnostic on S&P 100 (narrow - ~Lo 1991 finding).
- [x] Hurst sort on S&P 100 (S1).
- [x] Hurst sort on S&P 600 (small caps). Null.
- [x] Frac-diff as standalone cross-sectional factor (A3). Null.
- [x] S2 proper: FFD + momentum + vol features in LightGBM with 22-day
  embargoed walk-forward. Initial Sharpe 1.84 WAS LEAKAGE. After embargo
  fix: Sharpe 0.015 / -0.08 net. Null.
- [x] S4 intraday hourly seasonality. Open hour has 2.4x mean |return|.
  Classical microstructure reproduced. Execution alpha only.
- [x] **ETF cross-asset Hurst sort.** Sharpe -0.48 net, 95% CI just
  crosses zero. Composition analysis shows the sort is a long
  commodities+EM / short bonds+developed regime bet, not a fractal edge.
  Hurst proxies asset-class fundamentals.

Nearly pointless given accumulated null findings:
- [ ] S3 Lo-filtered trend universe.

Potentially salvageable but need new approaches:
- [ ] **Residualized Hurst** - regress H on asset class + vol,
  sort on the residual. Might isolate genuine fractal content from
  asset-class noise. This is the single remaining respectable directional
  test.
- [ ] **MF-DFA on VIX vs SPX realized vol** (A5) as a volatility regime
  indicator, feeding options strategies.
- [ ] Reality-check S4 intraday result with live broker execution data.
- [ ] Carr-Madan BS/Heston pricer (B1) - purely educational at this point.

## The two lessons to encode going forward

1. **Embargo every walk-forward CV.** The GBM embargo bug (Sharpe 1.84 ->
   0.015 after fix) is the single most valuable artifact here. Any future
   backtest in this repo must audit its embargo before celebrating a
   Sharpe.
2. **Compositional analysis on every sort.** The ETF sort's Sharpe -0.48
   was the *consequence* of an unintended long-commodities / short-bonds
   tilt. If we hadn't logged the composition, we might have concluded
   "Hurst has anti-alpha on ETFs" rather than "Hurst ranks ETFs by asset
   class." Always ask: *what is the sort actually holding?*

Updates are appended to [`RESULTS.md`](RESULTS.md).
