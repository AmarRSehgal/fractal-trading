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

## Summary so far

| Idea                                     | Status   | Edge?    |
|------------------------------------------|----------|----------|
| DFA / modified R/S estimators valid      | Done     | N/A      |
| FFD / d-picker valid                     | Done     | N/A      |
| S1 Hurst cross-sectional sort (S&P 100)  | Done     | Marginal-to-null |
| A3 FFD as factor (S&P 100)               | Done     | No       |
| S1 on Russell 2000                       | TODO     | Unknown  |
| S2 FFD + GBM features (proper test)      | TODO     | Unknown  |
| S3 Lo-filtered trend                     | TODO     | Unknown  |
| S4 Intraday seasonality                  | TODO     | Unknown  |
| A5 MF-DFA regime                         | TODO     | Unknown  |
| B1 Carr-Madan pricer                     | TODO     | Educational only |

**Honest state of play:** first directional test (S1) was weak. Retail
fractal alpha on liquid US stocks looks thin. The most likely wins from here
are (a) execution-timing improvements via A4/S4, and (b) frac-diff as one
feature in a larger model (S2), not a standalone factor.
