# fractal-trading

Personal research project: apply fractal methods (Hurst, DFA, Lo's modified
R/S, fractional differentiation, multifractal spectra) and Fourier /
wavelet methods to **US equity trading and analysis**.

## Why

Most retail material on "fractal trading" is narrative, not math. This repo
restricts itself to ideas that are (a) grounded in peer-reviewed literature
and (b) falsifiable with out-of-sample evaluation. It also refuses the most
common Fourier misuse - running FFT on raw price to find cycles.

## What's here

- [`IDEAS.md`](IDEAS.md) - concrete analysis and strategy ideas with formulas,
  pseudocode, and evaluation plans. Now contains first-pass empirical
  findings inline.
- [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) - per-paper notes with honest
  access status.
- [`RESULTS.md`](RESULTS.md) - raw numeric results from each experiment, in
  date order.
- `src/fractal_trading/` - library code
  - `hurst.py` - DFA and Lo's modified R/S (validated on fBm)
  - `mfdfa.py` - multifractal DFA (h(q) spectrum, Delta h)
  - `fracdiff.py` - fixed-width fractional differencing
  - `synthetic.py` - fBm / white noise / AR(1) generators for validation
  - `data.py` - yfinance loader with parquet cache
  - `universe.py` - S&P 100 / Dow 30 ticker lists
  - `backtest.py` - cross-sectional sort harness, i.i.d. and stationary
    block bootstrap CIs, paired Sharpe-difference CI, notional-traded
    transaction costs
- `scripts/` - runnable entry points (13 scripts, numbered)
- `tests/` - synthetic-series validation + leakage and cost guards
  (25 tests, all passing)
- `results/` - CSVs and PNGs from runs

## Quick start

```bash
# Use the real system python (MacPorts 3.13). NOTE: on this machine bare
# `python3` resolves to an unrelated work virtualenv and $PYTHONPATH is set
# to a work repo, so both must be bypassed:
#     alias fpy='env -u PYTHONPATH /opt/local/bin/python3.13'
env -u PYTHONPATH /opt/local/bin/python3.13 -m pip install --user \
    yfinance numpy pandas scipy statsmodels matplotlib lightgbm pyarrow

# run the test suite (includes embargo / lookahead guards)
env -u PYTHONPATH PYTHONPATH=src /opt/local/bin/python3.13 -m pytest tests/ -q

# validate estimators on synthetic data
python3 scripts/01_validate_estimators.py

# compute Hurst distribution over S&P 100 (writes CSV + PNG to results/)
python3 scripts/02_hurst_distribution.py --universe sp100

# backtest the Tier-1 Hurst sort strategy
python3 scripts/03_hurst_sort_backtest.py --universe sp100

# compare fractional-differencing factor vs 12-1 momentum
python3 scripts/04_fracdiff_comparison.py

# audit the estimator's own noise floor before trusting any H dispersion
python3 scripts/12_estimator_power.py --end 2026-08-31

# the baseline replication (Bariviera 2017, BTC time-varying Hurst)
python3 scripts/13_bariviera_btc.py --end 2026-08-31
```

**Reproducibility.** Every script takes `--end`; pass it to pin a run
(`--end 2026-04-20` reproduces the Round 1-3 numbers, `--end 2026-08-31`
the Round 4 re-run). Left unset, `end` defaults to *today*, so the sample
silently grows and numbers drift. Two things are still not pinned: package
versions (no lockfile) and the S&P constituent lists, which
`universe.py` scrapes live from Wikipedia - so a future clean checkout may
get a different universe than the one behind these results.

## Findings (audited 2026-09-01; data through 2026-08-28)

All net-of-cost Sharpes use 10bps per side with a 95% bootstrap CI, and all
of them reflect the **corrected** cost model - an earlier version
undercharged transaction costs by exactly 2x (see RESULTS.md §I).

| Experiment                             | Stat       | 95% CI         | Verdict |
|----------------------------------------|-----------:|----------------|---------|
| S1: Hurst sort, S&P 100                | Sh  0.25   | [-0.21, 0.68]  | Null    |
| S1: Hurst sort, S&P 600 (small caps)   | Sh -0.23   | [-0.77, 0.33]  | Null    |
| A3: FFD as standalone factor           | Sh -0.31   | wide           | Null    |
| S2: FFD + mom + vol in LightGBM        | Sh -0.07   | [-0.58, 0.41]  | Null    |
| S1: Hurst sort, ETFs (cross-asset)     | Sh -0.59   | [-1.19, -0.09] | Negative (regime bet) |
| Residualized Hurst sort, ETFs          | Sh -0.27   | [-0.81, 0.20]  | Null    |
| MF-DFA Δh VIX gate, **full-sample median** | +0.09 Δ | [-0.33, +0.50] | Null (and leaky - see below) |
| **MF-DFA Δh VIX gate, causal median**  | **-0.26 Δ**| [-0.58, +0.07] | **Negative** |
| Intraday execution: VWAP vs uniform    | -0.17 bps  | t = -1.0       | Null    |
| **Bariviera (2017) BTC replication**   | roll-H std 0.033 | noise floor 0.062 | **Null / era infeasible** |
| Baseline: 12-1 momentum                | Sh  0.07   | [-0.35, 0.52]  | Null    |

The S&P 600 row is pinned at `--end 2026-04-20` (the cached 603-ticker
sample); every other row runs through 2026-08-28.

Key findings:
1. **Narrow H dispersion in US equities.** S&P 100 full-sample mean
   H = 0.467, std 0.035. Only 3/99 reject Lo's null at 95% (chance).
2. **The mean H of 0.467 is real, not an estimator artifact.** Shuffling
   each stock's own returns (destroys time structure, keeps the fat-tailed
   marginal) gives DFA H = 0.498. US large caps are genuinely, mildly
   anti-persistent. This is the one positive empirical result in the repo -
   and it is not tradeable, see (3).
3. **The 500-day rolling Hurst factor is noise.** Its cross-sectional std
   (0.056) is *below* the estimator's own noise floor at n=500 (0.062), and
   its rank correlation across two DISJOINT 500-day windows is +0.04. Every
   Hurst-sort null in this repo is a sort on a factor with no test-retest
   reliability. The nulls are correct but uninformative about the
   hypothesis - the experiment was underpowered by construction.
4. **Leakage, not skill, drove GBM Sharpe 1.84.** 22-day embargo fix
   collapsed it to 0.015. The embargo is now correct and pinned by a test.
   With costs charged correctly it is -0.07.
5. **The VIX regime gate was leaky too.** It thresholded Δh on the
   *full-sample* median, so the 2008 gate depended on 2026 data. Replacing
   it with an expanding-window median drops the gate's Sharpe from 0.71 to
   0.36 - *below* buy-and-hold's 0.62. The repo's "best signal" was an
   artifact.
6. **ETF Hurst sort is a long-commodities/short-bonds regime bet in
   fractal clothing**, and after controlling for asset class and vol
   (R^2 = 0.996) the residual sort is null.
7. **Intraday volatility concentrates at market open** (2.4x average).
   Real microstructure finding, but "avoid open" execution is *worse*
   than uniform on mean cost.

8. **The baseline was never run, and cannot be.** Bariviera (2017) - the
   paper this whole program rests on - shows BTC persistent in 2011-2013.
   yfinance's BTC history starts 2014-09, so that era is unreachable here.
   On the era that is reachable, rolling 500-day H has std 0.033 against a
   0.062 noise floor: indistinguishable from no memory. The full-sample
   H of 0.562 *looks* persistent but no sub-period reproduces it and Lo's
   test does not reject - it is regime mixing, and it is exactly the
   artifact this repo would have reported as a finding in Round 1.
9. **Costs were undercharged 2x** in every net-of-cost number, because the
   cost base was a gross-normalized turnover ratio rather than notional
   traded. Fixing it moved the GBM from +0.02 to -0.07 and the FFD factor
   from ~0.01 to -0.31. High-turnover strategies were the most flattered.

**Interpretation.** Ten directional + execution tests plus the baseline
replication. **Zero** reject zero at 95%; the two that once looked marginal
were both leakage, and every net number was additionally flattered by a 2x
cost discount. The durable output is the **infrastructure** (bootstrap CIs
i.i.d. *and* block, a cost model that charges notional traded, embargoed
walk-forward, composition + residualization audits, and an estimator
noise-floor calibration) and five methodological lessons: audit the
embargo, audit *every* threshold for full-sample contamination, audit
composition, residualize before claiming edge, and **measure the
estimator's noise floor before believing any dispersion you sort on**.

The last one subsumes the rest. The 500-day DFA noise floor is 0.062 while
the true cross-sectional H dispersion is ~0.027, so no Hurst sort at this
window could ever have resolved the effect. That was computable on day one.

See `RESULTS.md` for full numbers and `IDEAS.md` for what's still worth doing.

## Caveats

- **Survivorship bias.** yfinance gives you currently-listed tickers only.
  Delistings are missing. All backtest Sharpes are biased upward, and the
  `dropna(thresh=...)` universe filter uses full-sample availability, which
  compounds it.
- **Estimator power, not just estimator bias.** DFA is unbiased here
  (bias <= 0.007 on fBm, and 0.498 on shuffled real returns), but its
  standard error is 0.062 at n=500 and 0.015 at n=5400. Any claim about
  cross-sectional H dispersion must be compared against that floor.
- **Costs are charged on notional traded** (`sum|dw|` per rebalance), not on
  a gross-normalized turnover ratio. The latter halves the charge on a
  dollar-neutral book; that was the bug through Round 4. `turnover` is still
  *reported* as a fraction of the gross book, so it is not the cost base -
  don't reintroduce it as one.
- **Multiple testing is not corrected for.** Ten experiments at a nominal
  95% level. Since none reject, this only makes the negative conclusion
  safer - but a future positive would need a correction.
- **Monthly L/S returns are non-overlapping**, so the i.i.d. bootstrap in
  `BacktestResult.bootstrap_sharpe_ci` is admissible for them. It is NOT
  admissible in general; use `sharpe_ci(..., expected_block=k)` for daily or
  overlapping series. One committed series (`hurst_ls_returns_sp100`) has
  significant serial correlation (Ljung-Box p = 0.006) and its i.i.d. CI
  excludes zero where the block-bootstrap CI does not.
- Fractional-differencing was tested both as a standalone factor and as one
  feature in a GBM (script 07). Both null.
