# RESULTS log

Raw numerical findings from each experiment. Append new entries at the bottom.

---

## 2026-04-20: Estimator validation (synthetic)

**Script:** `scripts/01_validate_estimators.py`

### DFA Hurst estimator

n = 4000, 20 seeds per true-H value.

| True H | DFA mean | DFA std | DFA min | DFA max |
|--------|----------|---------|---------|---------|
| 0.30   | 0.297    | 0.015   | 0.264   | 0.332   |
| 0.50   | 0.490    | 0.021   | 0.449   | 0.542   |
| 0.60   | 0.588    | 0.024   | 0.540   | 0.647   |
| 0.70   | 0.686    | 0.026   | 0.634   | 0.750   |
| 0.80   | 0.784    | 0.028   | 0.729   | 0.853   |

Conclusion: unbiased across the tested range. Std ~ 0.02-0.03 at n=4000;
expect wider CI on shorter windows.

### Modified R/S (Lo 1991)

n = 2000.

| Series        | Rejections at 95% | Comment                        |
|---------------|-------------------|--------------------------------|
| White noise   | 2/50              | correct size (~5% alpha)       |
| AR(1) phi=0.5 | 2/50              | no spurious rejection           |

Implied H from the statistic, AR(1) phi=0.5:
- Naive (q=0): 0.591 - spurious long memory
- Modified (Andrews q): 0.525 - correctly reined in

This reproduces Lo's key finding on synthetic AR(1): modified R/S does not
falsely flag short-range dependence as long memory.

---

## 2026-04-20: Hurst distribution on S&P 100

**Script:** `scripts/02_hurst_distribution.py --universe sp100`
**Period:** 2005-01-01 through 2026-04-20
**Universe:** 100 current S&P 100 tickers (survivorship-biased)

| Metric                        | Value |
|-------------------------------|-------|
| N stocks with >= 500 days     | 100   |
| DFA H mean                    | 0.465 |
| DFA H std                     | 0.034 |
| DFA H 5th percentile          | 0.415 |
| DFA H 95th percentile         | 0.517 |
| Modified R/S 95% rejections   | 5/100 |

Top 5 most persistent: AIG (0.55), NFLX (0.55), NVDA (0.55), AMD (0.53), GE (0.53)
Top 5 most anti-persistent: AVGO (0.40), MDLZ (0.40), ABT (0.41), DUK (0.41), ORCL (0.41)

**Interpretation.** This reproduces Lo (1991)'s finding on a modern data
set. Dispersion of H across S&P 100 is tight (range ~0.10 at 5-95
percentiles) and only 5 stocks reject the no-long-memory null at 95% -
approximately what you get by chance. Any cross-sectional H-based strategy
has very little raw material to work with.

---

## 2026-04-20: S1 - Cross-sectional Hurst quintile sort

**Script:** `scripts/03_hurst_sort_backtest.py --universe sp100 --lookback 500`
**Period:** 2005-01-01 through 2026-04-20 (212 monthly rebalances)
**Universe:** S&P 100, filtered to 91 names with >= 80% valid data
**Factor:** rolling 500-day DFA Hurst per stock, recomputed every 21 days
**Portfolio:** long top quintile of H, short bottom quintile, equal weight
**Costs modeled:** none (gross)

| Metric          | Value   |
|-----------------|---------|
| Annual return   | 4.92%   |
| Annual vol      | 10.5%   |
| Sharpe          | 0.47    |
| Max drawdown    | -28.1%  |
| Hit rate        | 51.4%   |
| Turnover / reb  | 42.3%   |

**Cost estimate:** with 42% monthly turnover at 10 bps round trip:
0.423 * 0.002 * 12 = 1.02% annual drag -> Sharpe ~0.37 after costs.
Survivorship haircut of ~1.5% annual return: Sharpe ~0.23.

**Conclusion.** This is a **marginal-to-null** result. Consistent with Lo
1991 - there is no meaningful long-memory structure to exploit in large-cap
US equities.

**Next experiments** (see IDEAS.md S1):
1. Rerun on Russell 2000 (wider H dispersion expected on small-caps).
2. Longer lookback (3-year) to reduce factor noise.
3. Quarterly rebalance to cut turnover by ~3x.

---

## 2026-04-20: A3 test - Fractional differencing as cross-sectional factor

**Script:** `scripts/04_fracdiff_comparison.py --d 0.4`
**Period:** 2005-01-01 through 2026-04-20
**Universe:** S&P 100 (survivorship-biased)

| Factor          | Sharpe | Ann Ret | Ann Vol | Max DD  | Turnover |
|-----------------|--------|---------|---------|---------|----------|
| Momentum 12-1   | 0.171  | 3.1%    | 17.9%   | -51.4%  | 37.7%    |
| FFD z-score d=0.4 | 0.006 | 0.1%    | 11.6%   | -38.8%  | 73.9%    |
| Combined (50/50)| -0.015 | -0.2%   | 12.8%   | -34.3%  | 53.1%    |

**Conclusion.** FFD-as-standalone-factor fails here. **Turnover is nearly
2x momentum's** (my prior was that FFD would be lower turnover - wrong).

**Important caveat.** This does NOT invalidate López de Prado's original
proposal. FFD was proposed as an **ML feature** alongside other features in
a nonlinear model, not as a standalone cross-sectional z-score. That
experiment (S2 in IDEAS.md) has not been run yet.

---

## 2026-04-20 (second pass): S1 extended to S&P 600 (small caps)

**Script:** `scripts/05_hurst_sort_sp600.py`
**Period:** 2010-01-01 through 2026-04-20 (156 monthly rebalances)
**Universe:** S&P 600, filtered to 480 names with >= 60% data coverage
**Factor:** rolling 500-day DFA Hurst, 21-day step
**Portfolio:** quintile long-short, min 20 names per leg

### H distribution

| Metric | Value |
|--------|-------|
| Mean H | 0.471 |
| Std    | 0.033 |
| p05-p95| [0.418, 0.518] |

**Finding.** S&P 600 H distribution is nearly identical to S&P 100
(0.465, 0.034). My hypothesis that small-caps would have wider H dispersion
was wrong. The dispersion ceiling on liquid US stocks (large or small) is
~0.10 at 5-95 percentiles. Lo (1991) holds on small-caps too.

### Backtest with bootstrap CI and transaction costs

| Metric            | Gross    | Net (10bps/side) |
|-------------------|----------|------------------|
| Annual return     | -1.1%    | -1.7%            |
| Annual vol        | 7.9%     | 7.9%             |
| Sharpe            | -0.14    | -0.22            |
| Max drawdown      | -24.2%   | -28.0%           |
| Hit rate          | 46.2%    | 44.2%            |
| Turnover / rebal  | 27.9%    | 27.9%            |
| Sharpe 95% CI     | [-0.69, 0.42] | [-0.78, 0.33] |
| N monthly rebals  | 156      | 156              |

**Conclusion.** S1 on small-caps is a **clean null**. Point estimate
slightly negative, 95% CI straddles zero. The Hurst sort hypothesis is
dead on liquid US equities, large or small. Do not pursue further without
a fundamentally different data source (intraday microstructure H, or
cross-country H, etc.).

---

## 2026-04-20: S4 - Intraday seasonality (hourly bars)

**Script:** `scripts/06_intraday_seasonality.py`
**Period:** last 700 days (2024-05 to 2026-04), 1-hour bars
**Universe:** 26 liquid stocks + 4 index ETFs

### Finding: volatility front-loads dramatically

Normalized |log return| per hour-of-day bucket (UTC, across DST/EST mixing):

| Hour UTC | |Return| norm | Volume norm |
|----------|---------------|-------------|
| 13       | 2.43x         | 0.96x       |
| 14       | 1.48x         | 1.00x       |
| 15       | 0.83x         | 1.00x       |
| 16       | 0.75x         | 1.00x       |
| 17       | 0.68x         | 1.00x       |
| 18       | 0.61x         | 1.00x       |
| 19       | 0.62x         | 1.00x       |
| 20       | 0.60x         | 1.06x       |

**Caveats.**
- Hour bucketing mixes EST and EDT over 2 years; results are approximate.
  Hour 13 UTC captures the open in EDT months (9:30 ET), later sessions
  captured in adjacent buckets during EST.
- 1-hour bars flatten the closing-auction spike. To see a classic U-shape
  you need 15-minute bars (unavailable on yfinance beyond 60 days).

**Takeaway.** Volatility at the US market open is ~2.4x the intraday
average. Classical microstructure lit. reproduced. **Actionable
consequence:** for any discretionary or systematic execution:
- Avoid passive posting during the open hour unless you want inventory
  risk.
- Concentrate passive fills in the 15:00-20:00 UTC band where volatility
  is below average and volume is at/above average.

Volume seasonality is nearly flat when normalized per-stock - suggests
yfinance hourly volume is either smoothed or the closing-auction volume
isn't captured in hour-bar granularity.

---

## 2026-04-20: S2 - GBM walk-forward (FFD + momentum + vol features)

**Script:** `scripts/07_gbm_walkforward.py`
**Period:** 2005-01 through 2026-04, monthly retraining
**Universe:** S&P 100 (91 survivors)
**Features per stock per date:** `mom_12_1`, `ret_1m`, `vol_60`, `vol_252`,
 `fd_z` (frac-diff z-score, d=0.4)
**Model:** LightGBM regressor, trailing 5-year training window, 100 trees
**Target:** next-21-day log return
**Portfolio:** quintile long-short by predicted return

### FIRST RUN (WITHOUT EMBARGO) - LEAKAGE

| Metric           | Value |
|------------------|-------|
| Sharpe (gross)   | 1.84  |
| Sharpe (net 10bps)| 1.75 |
| 95% CI (net)     | [1.33, 2.20] |
| Hit rate         | 73%   |

**This was too good to be true.** Checked the walk-forward window:

At rebalance date `d`, training rows at `d_train \in [d - 21, d)` have
targets spanning `[d_train, d_train + 21]`, which extends PAST `d` into
the test target window. Those forward returns are NOT observed at test
time - pure lookahead.

### FIXED RUN (WITH 22-BDAY EMBARGO)

Training rows restricted to `d_train <= d - 22 BDay`, so no training
target overlaps the test period.

| Metric           | Gross   | Net (10bps/side) |
|------------------|---------|------------------|
| Annual return    | 0.2%    | -1.2%            |
| Annual vol       | 13.8%   | 13.8%            |
| Sharpe           | 0.015   | -0.08            |
| Max drawdown     | -34.5%  | -42.1%           |
| Hit rate         | 49.1%   | 48.0%            |
| 95% CI Sharpe    | [-0.48, 0.54] | [-0.58, 0.43] |

**Conclusion.** With honest walk-forward, the GBM model adds **zero** edge.
Sharpe dropped from 1.84 to 0.015 - a 100x reduction - from a single
embargo fix. This is a textbook demonstration of why careful leakage
analysis matters.

### Full-sample feature importance (for context only)

| Feature      | LGBM gain | Rank |
|--------------|-----------|------|
| vol_252      | 824       | 1    |
| mom_12_1     | 675       | 2    |
| vol_60       | 663       | 3    |
| ret_1m       | 329       | 4    |
| fd_z         | 309       | 5    |

Frac-diff is the **least** important feature per LGBM gain. This
invalidates the López de Prado motivation on this particular feature set
and universe. (Caveat: feature importance is gain-based and can be
misleading; fd_z is multicollinear with other price-level features.)

### Baseline (Momentum 12-1 alone)

| Metric           | Gross   | Net (10bps)      |
|------------------|---------|------------------|
| Sharpe           | 0.17    | 0.14             |
| 95% CI           | [-0.25, 0.68] | [-0.27, 0.64] |

Momentum on S&P 100 also fails to reject zero Sharpe at 95% CI. That is
consistent with the known collapse of momentum in large-cap US equities
post-2000.

---

## Summary - four tests done

| Experiment                                   | Edge?  | Sharpe (net)   | 95% CI        |
|----------------------------------------------|--------|----------------|---------------|
| S1 Hurst sort S&P 100                        | ~Null  | 0.23 (est.)    | Wide, likely crosses 0 |
| S1 Hurst sort S&P 600 (small caps)           | Null   | -0.22          | [-0.78, 0.33] |
| A3 FFD as standalone factor S&P 100          | Null   | 0.01           | Wide          |
| S2 GBM w/ FFD+mom+vol features, embargoed    | Null   | -0.08          | [-0.58, 0.43] |
| S4 Intraday seasonality (execution context)  | Real   | N/A            | Real ~2.4x open vol |

| Idea                                         | Status | Edge?   |
|----------------------------------------------|--------|---------|
| DFA / modified R/S estimators valid          | Done   | N/A     |
| FFD / d-picker valid                         | Done   | N/A     |
| Bootstrap CI + transaction costs in backtest | Done   | N/A     |
| S1 Hurst sort S&P 100                        | Done   | Weak    |
| S1 Hurst sort S&P 600                        | Done   | Null    |
| A3 FFD as standalone factor                  | Done   | Null    |
| S2 FFD + GBM features                        | Done   | Null    |
| S4 Intraday hourly seasonality               | Done   | Real execution signal |
| S3 Lo-filtered trend universe                | TODO   | Unknown |
| A5 MF-DFA regime                             | TODO   | Unknown |
| B1 Carr-Madan pricer                         | TODO   | Educational |

---

## 2026-04-20: ETF cross-asset Hurst sort (last directional idea)

**Script:** `scripts/08_etf_hurst_sort.py`
**Period:** 2008-01 through 2026-04 (182 monthly rebalances)
**Universe:** 63 ETFs (equity sectors, US/intl/EM equities, bonds,
 commodities, FX, REITs) with curated asset-class tags

### H distribution by broad asset class

| Class        | N  | Mean H | Std H |
|--------------|----|--------|-------|
| commodity    | 6  | 0.519  | 0.052 |
| international| 17 | 0.512  | 0.017 |
| fx           | 2  | 0.503  | 0.006 |
| em           | 9  | 0.502  | 0.026 |
| bonds        | 8  | 0.502  | 0.033 |
| reit         | 2  | 0.498  | 0.002 |
| us (equity)  | 18 | 0.488  | 0.022 |
| preferred    | 1  | 0.476  | -     |

**Dispersion is still narrow.** Commodity is the widest at std 0.052 but
most classes sit in 0.02-0.03, similar to the individual-stock result.
Commodities and international DM lean slightly persistent (H ~0.51-0.52);
US equities lean slightly anti-persistent (H ~0.49). Differences are
small and dominated by asset-class fundamentals, not fractal memory.

### Backtest

| Metric            | Gross    | Net (10bps/side) |
|-------------------|----------|------------------|
| Annual return     | -4.1%    | -4.8%            |
| Annual vol        | 9.8%     | 9.8%             |
| Sharpe            | -0.42    | -0.48            |
| Max drawdown      | -58.9%   | -62.5%           |
| Hit rate          | 42.9%    | 42.9%            |
| Turnover / rebal  | 27.5%    | 27.5%            |
| Sharpe 95% CI     | [-0.98, 0.09] | [-1.06, 0.03] |
| N monthly rebals  | 182      | 182              |

Point estimate is materially negative though the 95% CI just crosses zero.
**Worse than null - the sort is systematically losing money.**

### Why: composition analysis

The `holdings` DataFrame tells us what the sort is actually buying and
selling over 182 months.

**Most frequent longs** (top of H distribution):

| Ticker | Asset class       | % of rebals in long leg |
|--------|-------------------|-------------------------|
| USO    | commodity_oil     | 52%                     |
| EMB    | bonds_em          | 51%                     |
| EWZ    | em_brazil         | 51%                     |
| EWS    | intl_singapore    | 48%                     |
| DBC    | commodity_broad   | 45%                     |
| SLV    | commodity_silver  | 43%                     |
| EWH    | intl_hongkong     | 42%                     |
| LQD    | bonds_ig_corp     | 42%                     |
| TUR    | em_turkey         | 41%                     |
| DBA    | commodity_ag      | 38%                     |

**Most frequent shorts** (bottom of H distribution):

| Ticker | Asset class       | % of rebals in short leg |
|--------|-------------------|--------------------------|
| EZA    | em_safrica        | 66%                      |
| EWU    | intl_uk           | 32%                      |
| TIP    | bonds_tips        | 31%                      |
| HYG    | bonds_hy_corp     | 31%                      |
| SHY    | bonds_sh_treasury | 31%                      |
| EWA    | intl_australia    | 30%                      |
| FXI    | em_china          | 30%                      |
| XLP    | us_sector_staples | 30%                      |
| TLT    | bonds_lg_treasury | 29%                      |
| EWJ    | intl_japan        | 27%                      |

### The finding

This is not a fractal strategy. It is a **macro regime bet in disguise**:
**long commodities + EM / short bonds + developed equities**. That is
the wrong side of the 2010-2026 macro trade. Commodities had a lost
decade (oil -75% 2014-16, -60% in 2020); bonds rallied until 2022; US and
developed equities compounded ~12% annual. Running this sort for 15 years
was an implicit short position on the dominant post-GFC regime.

**Hurst exponent does not purely measure long memory on cross-asset
instruments.** It picks up asset-class fundamentals: how fast the price
mean-reverts around its drift is dominated by carry, term premium,
volatility clustering, and liquidity - not long-range dependence.

### Conclusion

Even on the most diversified, highest-dispersion universe we tried, the
cross-sectional Hurst sort generates negative alpha. The mechanism is
clear from composition analysis: Hurst rankings proxy asset-class mix,
and the resulting portfolio is an asset-class bet without style-timing.

Any future strategy in this repo should **control for asset-class means**
before using Hurst as a signal - or, more practically, **stop using
raw Hurst as a ranking signal** and instead use Hurst residuals after
regressing out asset-class and volatility.

---

## Summary - all five directional tests done

| Experiment                                      | Sharpe net | 95% CI          | Verdict |
|-------------------------------------------------|-----------:|-----------------|---------|
| S1 Hurst sort S&P 100                           | ~0.23 est  | wide            | Weak    |
| S1 Hurst sort S&P 600 (small caps)              | -0.22      | [-0.78, 0.33]   | Null    |
| A3 FFD as standalone factor S&P 100             |  0.01      | wide            | Null    |
| S2 GBM w/ FFD+mom+vol embargoed                 | -0.08      | [-0.58, 0.43]   | Null    |
| **S1 Hurst sort ETFs (cross-asset)**            | **-0.48**  | [-1.06, 0.03]   | **Negative (regime bet)** |
| S4 Intraday seasonality (execution diagnostic)  | N/A        | -               | Real signal |
| Baseline momentum 12-1                          |  0.14      | [-0.27, 0.64]   | Weak    |

## Honest state of play

**Five directional fractal tests on retail-accessible US instruments.
Four nulls, one structural negative.** The last test (ETF cross-asset)
was the strongest candidate because of wider H dispersion across asset
classes; it still failed, and the composition analysis shows *why* -
Hurst correlates with asset-class fundamentals more than with genuine
long memory.

**Durable wins from this project:**
1. Validated DFA / modified R/S / FFD estimators in `src/fractal_trading/`.
2. Cross-sectional backtest harness with cost model, bootstrap CIs, and
   the embargo fix.
3. The intraday seasonality signal at market open (real, reproducible).
4. The embargo-catch GBM result (Sharpe 1.84 -> 0.015 after fix) as a
   worked example of why walk-forward leakage analysis matters.
5. Composition analysis pattern (what is the sort actually holding?) -
   essential for any future factor strategy to avoid "alpha in name only."

---

## 2026-04-20 (third pass): residualized Hurst + MF-DFA VIX + execution

Three experiments left on the roadmap after the ETF sort. Running all of
them here for completeness.

### Experiment A: Residualized Hurst sort on ETFs (script 09)

**Hypothesis.** The plain ETF Hurst sort was an asset-class bet in
disguise (script 08). If we regress H on asset class + vol at each
rebalance and use the residual as the sort factor, we isolate the
"genuine" fractal content.

**Diagnostic finding.** At each rebalance date, regress
`H ~ C(broad_class) + vol`. The mean R^2 over the last 20 rebalances is
**0.996** - essentially all cross-sectional variation in H across
retail-accessible ETFs is explained by asset class and volatility. The
"fractal information" is less than 0.4% of the total variance.

**Backtest (residualized H sort, 63 ETFs, 2008-2026, 182 months):**

| Metric            | Plain H   | Residualized |
|-------------------|-----------|--------------|
| Sharpe gross      | -0.42     | -0.05        |
| Sharpe net 10bps  | -0.48     | -0.13        |
| 95% CI net Sharpe | [-1.06, 0.03] | [-0.65, 0.38] |
| Max DD            | -62%      | -23%         |
| Vol               | 9.8%      | 7.5%         |
| Turnover/rebal    | 27.5%     | 24.8%        |

**Finding.** Residualization moves Sharpe from -0.48 toward 0 (to -0.13,
CI straddles zero). The asset-class-adjusted residual has no
predictive edge - consistent with R^2 = 0.996 telling us there is
almost nothing to predict. Drawdown also falls sharply because the
portfolio is no longer a concentrated asset-class tilt.

Residualized longs are now spread: bonds_em, intl_singapore,
intl_hongkong, em_turkey, bonds_ig_corp, em_brazil. Shorts include
em_safrica, intl_developed, bonds, commodity_natgas. Still biased but
much less than plain.

**Conclusion.** After controlling for obvious confounds, Hurst contains
no meaningful cross-sectional signal on retail ETFs. This closes the
last directional fractal hypothesis on retail-accessible instruments.

---

### Experiment B: MF-DFA on VIX as regime indicator (script 10)

**Implementation.** MF-DFA with q in [-4, ..., +4]; validated on
synthetic series (monofractal WN and fBm recover constant h(q);
skew-t distributed shows mild q-dependence). See `tests/test_mfdfa.py`.

**Full-sample VIX log returns (2005-2026):**

| Metric | Value |
|--------|-------|
| h(q=-4)   | 0.416 |
| h(q=-2)   | 0.391 |
| h(q= 0.5) | 0.354 |
| h(q= 2)   | 0.329 |  (classical Hurst - mean-reverting, expected for VIX)
| h(q= 4)   | 0.292 |
| Delta h   | 0.124 |

VIX has genuine but mild multifractality: Δh = 0.12 is above the Δh
of fBm (<0.1) but far below a binomial cascade. Classical H = 0.33
confirms VIX strongly mean-reverts.

**Rolling Δh (500-day window, 232 rolling points) vs 21-day-forward
outcomes:**

| Signal     | vs SPY fwd ret | vs VIX fwd ret | vs SPY fwd vol | vs VIX level |
|------------|----------------|----------------|----------------|--------------|
| Delta h    | -0.119         | +0.035         | -0.007         | -0.172       |
| h(q=2)     | +0.093         | -0.050         | +0.113         | +0.334       |

Δh vs SPY forward return at -0.12 is the single interesting correlation
- wider multifractality modestly predicts negative SPY returns.

**Naive regime gate.** Hold SPY when Δh is below its median; flat
otherwise.

| Strategy           | Ann ret | Ann vol | Sharpe | Time in mkt |
|--------------------|---------|---------|--------|-------------|
| Buy-and-hold SPY   | +10.0%  | 16.8%   | 0.59   | 100%        |
| Δh regime gate     | +8.1%   | 11.6%   | 0.70   | 50%         |

The gate improves Sharpe by +0.11 - but is this statistically real?

**Bootstrap test** (5000 resamples):
- Mean Sharpe diff (gate - BH): +0.097
- 95% CI: [-0.298, +0.506]
- P(diff > 0): 67.5%

**Not statistically significant.** The gate fails to reject zero. Point
estimate tilted positive but indistinguishable from noise with this
sample size.

**Conclusion.** MF-DFA on VIX produces the most promising fractal-derived
signal seen in this project, but it is not statistically significant at
95% on 21 years of data. Not trade-ready.

---

### Experiment C: Intraday execution cost backtest (script 11)

**Setup.** 26 liquid US stocks, last 700 days of 1-hour bars. Fit volume
profile on first half; simulate execution strategies on second half.
6,110 (day x ticker) observations.

**Strategies:** uniform (equal weight across bars), vwap_profile
(proportional to fitted hourly volume share), avoid_open (uniform but
skip first bar - motivated by script 06 finding that open has 2.4x
avg volatility), front_load, end_load.

**Implementation shortfall vs day VWAP (bps):**

| Strategy       | Mean  | Median | Std   | p10    | p90   |
|----------------|-------|--------|-------|--------|-------|
| uniform        | +1.25 | +1.40  | 20.3  | -18.5  | +21.5 |
| vwap_profile   | +1.09 | +1.07  | 16.1  | -13.7  | +16.5 |
| avoid_open     | +1.49 | +1.81  | 29.7  | -28.6  | +31.3 |
| front_load     | -0.17 | -2.14  | 60.1  | -62.6  | +62.3 |
| end_load       | +1.34 | +2.26  | 62.2  | -65.5  | +67.9 |

**Pairwise tests:**

- vwap_profile vs uniform: mean diff -0.17 bps, t-stat -1.0 (not sig)
- avoid_open vs uniform: mean diff **+0.24 bps**, t-stat +1.5 (marginally
  worse than uniform, contrary to hypothesis)

**Key findings.**
1. **"Avoid open" is wrong on cost, even though open is high-vol.** High
   volatility at the open cuts both ways; average cost is no better than
   including it. The intuition from script 06 (2.4x volatility) translates
   to higher variance of execution cost, NOT lower mean cost.
2. **VWAP-profile reduces variance by ~20%** (std 16 bps vs uniform 20
   bps) but mean savings is negligible. Still useful if you want more
   predictable execution costs.
3. **Front-load has best mean (-0.17 bps)** but triple the variance of
   uniform. Unreliable.
4. **Per-ticker heterogeneity is large.** AMD and AAPL benefit from
   avoid-open (tech names with noisy opens). Defensive names (XOM, JNJ,
   WMT) get worse execution by avoiding open.

**Conclusion.** Execution alpha on retail-accessible liquid US equities
is smaller than I expected. The timing improvements are in the
0.1-0.3 bps range and noisy. For high-frequency or active day-trading
this would accumulate; for typical swing/position trading with 1-10
trades/day, execution timing is second-order to everything else.

---

## FINAL summary - eight tests, all done

| Experiment                                      | Sharpe/stat   | 95% CI           | Verdict |
|-------------------------------------------------|---------------|------------------|---------|
| S1 Hurst sort S&P 100                           | ~0.23 net     | wide             | Weak    |
| S1 Hurst sort S&P 600 (small caps)              | -0.22 net     | [-0.78, 0.33]    | Null    |
| A3 FFD as standalone factor S&P 100             |  0.01 net     | wide             | Null    |
| S2 GBM w/ FFD+mom+vol embargoed                 | -0.08 net     | [-0.58, 0.43]    | Null    |
| S1 Hurst sort ETFs (cross-asset)                | -0.48 net     | [-1.06, 0.03]    | Negative|
| **Residualized H sort ETFs**                    | **-0.13 net** | [-0.65, 0.38]    | **Null**|
| **MF-DFA VIX Delta h regime gate**              | +0.10 vs BH   | [-0.30, +0.51]   | Marginal (ns) |
| **Intraday execution (VWAP vs uniform)**        | -0.17 bps     | t=-1.0           | **Null** |
| Baseline momentum 12-1                          | +0.14 net     | [-0.27, 0.64]    | Weak    |

## The broader conclusion

**Eight directional or execution-related tests across individual stocks,
ETFs, and VIX. Zero are statistically significant positive at 95%.** Two
are marginally positive (VIX regime gate, intraday front-load) with
wide CIs; the rest are null or negative.

**What this project has actually produced:**
1. A validated library for DFA, modified R/S, FFD, MF-DFA on synthetic
   and real data.
2. A cross-sectional backtest harness with cost model, bootstrap CIs,
   embargoed walk-forward, and composition analysis.
3. A reproducible demonstration that **retail fractal alpha on
   yfinance-accessible instruments is not present to a statistically
   detectable degree**.
4. Three distinct methodological artifacts worth keeping:
   - Embargo audit (GBM Sharpe 1.84 -> 0.015 after fix)
   - Composition audit (ETF Sharpe -0.48 was a commodity/EM bet)
   - Residualization (R^2 0.996 showing how little H adds on top of
     asset class + vol)
5. The one borderline positive is the VIX MF-DFA regime gate; its point
   estimate is positive but the 95% CI straddles zero with 231 obs.

If this were a professional shop with paid data (delisting-aware,
minute-bar going back 20 years, options data), the next logical step
would be:
- re-run the Hurst sort with a bias-free universe and see if the
  survivorship haircut was masking real weak alpha (unlikely but
  possible)
- apply the VIX regime gate at higher frequency with options as the
  trading vehicle
- run MF-DFA on realized volatility for options vol-of-vol trades

On yfinance and retail access, this is as far as the methodology can go.

---

# Round 4 - 2026-08-31: methodology audit + out-of-sample extension

Re-ran everything on data through **2026-08-28** (4 months of genuinely
out-of-sample data past the Round 3 cutoff of 2026-04-17) and audited the
machinery that produces the nulls. Interpreter: MacPorts python 3.13.12,
numpy 2.4.3, pandas 2.3.3, statsmodels 0.15.0.

## A. Two bugs that made scripts unrunnable

`BacktestResult.stats` became a *method* (cost-aware) in Round 2, but
scripts 03 and 04 still called it as a `@property`. Script 03 raised
`AttributeError: 'function' object has no attribute 'items'`; script 04
crashed at `s['sharpe']`. Both had been dead since Round 2 - the Round 2/3
numbers for those experiments came from the older code. Fixed; both now
route through `report()` and take `--cost_bps`.

## B. LOOKAHEAD in the VIX regime gate (the repo's "best signal")

Script 10 thresholded Δh on `df["delta_h"].median()` - the **full-sample**
median over 2005-2026. The gate's 2008 decisions depended on 2026 data.

Replacing it with an expanding-window median (only rebalances already
observed, 40-rebalance warm-up) - data through 2026-08-28, n=235:

| Variant                      | Ann ret | Ann vol | Sharpe | Time in mkt |
|------------------------------|---------|---------|--------|-------------|
| Buy-and-hold SPY             | +10.5%  | 16.8%   | 0.62   | 100%        |
| Gate, full-sample median (leaky) | +8.2% | 11.5% | 0.71   | 50%         |
| Gate, expanding median (causal)  | +4.7% | 13.0% | **0.36** | 51%      |

Paired bootstrap of Sharpe(gate) - Sharpe(buy-and-hold), 5000 resamples:

| Variant | Bootstrap | Diff   | 95% CI            | P(diff>0) |
|---------|-----------|--------|-------------------|-----------|
| leaky   | i.i.d.    | +0.087 | [-0.328, +0.499]  | 64.6%     |
| leaky   | block(4)  | +0.087 | [-0.293, +0.436]  | 65.8%     |
| causal  | i.i.d.    | -0.261 | [-0.582, +0.067]  |  6.1%     |
| causal  | block(4)  | -0.261 | [-0.567, +0.051]  |  5.6%     |

**The one signal in this repo that pointed somewhere was an artifact.**
Made causal it does not merely fail to reject zero - it points negative.

Two secondary notes on this script. The bootstrap that produced the
Round 3 headline CI was never committed; it lives in the script now. And
the observations are non-overlapping *by construction* (the rolling step
of 21 days exactly equals the 21-day forward-return horizon), which is
why the block bootstrap does not widen the CI here.

## C. Bootstrap CI construction

The audit question was whether the i.i.d. bootstrap understates CI width
by ignoring serial correlation in overlapping returns. Findings:

- **The premise mostly does not apply.** `cross_sectional_sort_backtest`
  produces *non-overlapping monthly* L/S returns, for which i.i.d.
  resampling is admissible. Measured |lag-1 autocorrelation| <= 0.15 on
  all nine committed return series.
- **But the docstring's blanket reassurance is wrong in one live case.**
  `hurst_ls_returns_sp100` has lag-1 autocorrelation +0.15 and Ljung-Box
  p = 0.006. Its i.i.d. CI is [+0.03, +0.92] - excluding zero - while the
  stationary-block CI is [-0.12, +1.03], which includes it. The one
  experiment the README called "Weak" rather than "Null" was the one
  where the bootstrap choice changed the verdict.
- Added `stationary_bootstrap_indices`, `sharpe_ci(expected_block=...)`
  and `paired_sharpe_diff_ci` to `backtest.py`. The paired version matters
  for gate-vs-benchmark claims: the two series share the same underlying
  SPY returns, so an unpaired bootstrap would destroy their correlation.
- `tests/test_embargo.py` includes a synthetic case proving the mechanism
  is real when it *does* apply: on 21-day overlapping returns the block CI
  is >1.5x the width of the i.i.d. CI.

## D. Embargo: correct

Verified by reading and by test. A training row at `d_train` has target
over `[d_train, d_train + 21 trading days]`; `walk_forward_predict`
trains on `date <= date - BDay(22)`, so the last usable label closes one
business day before the test date. No leak.
`test_embargo_excludes_unobservable_training_targets` runs the real
function with a stub model and asserts the >= 21-business-day gap at
every rebalance it predicts for.
`test_cross_sectional_sort_has_no_lookahead` feeds the sort harness an
oracle factor equal to the *current* month's return and asserts it earns
nothing - it fails loudly if anyone removes the `shift(1)`.

## E. Estimator validation: unbiased, but far too noisy to sort on

DFA recovers known H well (200 seeds per cell):

| n    | true H | mean  | bias   | std   |
|------|--------|-------|--------|-------|
| 500  | 0.50   | 0.497 | -0.003 | 0.062 |
| 1000 | 0.50   | 0.496 | -0.004 | 0.045 |
| 2000 | 0.50   | 0.497 | -0.003 | 0.032 |
| 5300 | 0.50   | 0.500 | -0.000 | 0.026 |

So the nulls are real nulls, not artifacts of a biased estimator. **But
the standard error is the story.** At the 500-day lookback every backtest
in this repo uses, the estimator's own noise is std 0.062, while the
observed cross-sectional dispersion of the rolling factor is **0.056** -
*below the noise floor*. Rank persistence of the factor confirms it:

| lag (rebalances) | window overlap | Spearman rho |
|------------------|----------------|--------------|
| 1  (21d)   | 96%       | +0.866 |
| 6  (126d)  | 75%       | +0.674 |
| 12 (252d)  | 50%       | +0.426 |
| **24 (504d)** | **0% (disjoint)** | **+0.042** |
| 36 (756d)  | 0%        | +0.039 |
| 48 (1008d) | 0%        | +0.089 |

The apparent persistence at short lags is *shared data*, not a stable
stock property. Across two disjoint 500-day windows the ranking is
uncorrelated. **The Hurst sorts were sorting on noise.** That does not
change any verdict, but it changes the meaning: those nulls are not
evidence against cross-sectional long memory, they are evidence the
experiment could not have detected it at this window length.

## F. Shuffle control: the H = 0.467 result is real

Is mean H = 0.467 genuine anti-persistence, or DFA reacting to fat tails
and volatility clustering? Shuffling each stock's own returns destroys all
temporal structure (true H = 0.5) while preserving the marginal:

| series (S&P 100, n=99, full sample) | mean H | std   |
|-------------------------------------|--------|-------|
| real returns                        | 0.4668 | 0.0353 |
| shuffled returns                    | 0.4982 | 0.0147 |

DFA reads 0.498 on the shuffled data, so it is unbiased for this data's
distribution and the -0.031 gap in real data is genuine temporal
structure. This also gives a data-native noise floor of 0.0147 at full
sample, implying a true cross-sectional H dispersion of ~0.032 - real,
but only about half the 500-day estimator noise, which is exactly why
nothing survives at tradeable horizons.

## G. Out-of-sample extension to 2026-08-28

Nothing improved; the point estimates drifted slightly further toward
zero or negative.

| Experiment | Round 3 (to 2026-04-17) | Round 4 (to 2026-08-28) |
|------------|-------------------------|-------------------------|
| Hurst sort S&P 100, net Sharpe | 0.41, CI [-0.02, 0.86] | **0.31, CI [-0.14, 0.75]** |
| ETF plain H sort, net Sharpe   | -0.48, CI [-1.06, 0.03] | **-0.53, CI [-1.11, -0.03]** |
| Residualized ETF, net Sharpe   | -0.13, CI [-0.65, 0.38] | **-0.19, CI [-0.72, 0.28]** |
| GBM embargoed, net Sharpe      | 0.015 | **0.024, CI [-0.48, 0.52]** |
| Momentum 12-1, net Sharpe      | 0.14, CI [-0.27, 0.64] | **0.11, CI [-0.32, 0.56]** |
| VIX gate (leaky) diff          | +0.102 | **+0.087** |
| VIX gate (causal) diff         | n/a    | **-0.261** |
| S&P 100 mean H                 | 0.470 | **0.467, std 0.035, 3/99 reject** |

Note the S&P 100 Hurst sort: at the April cutoff its *gross* i.i.d. CI
was [+0.04, +0.92], excluding zero. Four more months plus the correct
block bootstrap both push it back across zero. It was never a signal.

## H. Reproduction status

Every Round 1-3 headline number reproduces when the run is pinned with
`--end 2026-04-20`. The Round 3 VIX bootstrap reproduces to within
resampling noise (+0.102 / [-0.307, +0.522] / 67.6% here vs the recorded
+0.097 / [-0.298, +0.506] / 67.5%).

Remaining reproducibility gaps: no dependency lockfile, and
`universe.py` scrapes S&P constituents live from Wikipedia, so the
universe (and therefore the parquet cache key) can drift under a future
clean checkout. `.data_cache/` is gitignored, so a clean checkout refetches
- verified working against yfinance 1.7.0 on 2026-08-31.

## Verdict after Round 4

**Nine tests, zero rejections, and both former near-misses were leakage.**
The Bariviera replication named in `RESEARCH_NOTES.md` as "a direct first
notebook" was never done - and it is a *crypto* result, while every
experiment here ran on US equities, the asset class with the weakest prior
for H != 0.5. The program tested its hypothesis where it was least likely
to hold, and did so with an estimator whose 500-day noise floor exceeded
the effect it was looking for.

This is a completed negative result. Write it up and stop.
