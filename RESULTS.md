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

If further work happens on this repo, pivot away from directional fractal
strategies on retail instruments. Three genuinely open questions remain:

- **S4 intraday execution** scaled into a real trading workflow.
- **Residualized Hurst**: regress H on asset-class + vol, use the
  residual as the sorting factor. Might isolate genuine fractal content
  from asset-class noise.
- **MF-DFA on VIX / realized vol** as a regime indicator for option
  strategies - still untested.
