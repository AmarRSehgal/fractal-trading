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
  - `fracdiff.py` - fixed-width fractional differencing
  - `synthetic.py` - fBm / white noise / AR(1) generators for validation
  - `data.py` - yfinance loader with parquet cache
  - `universe.py` - S&P 100 / Dow 30 ticker lists
  - `backtest.py` - cross-sectional sort backtest harness
- `scripts/` - runnable entry points (4 scripts, numbered)
- `tests/` - synthetic-series validation (10 tests, all passing)
- `results/` - CSVs and PNGs from runs

## Quick start

```bash
# use system python3 (not a venv) per personal-project convention
pip3 install --user yfinance numpy pandas scipy statsmodels matplotlib

# validate estimators on synthetic data
python3 scripts/01_validate_estimators.py

# compute Hurst distribution over S&P 100 (writes CSV + PNG to results/)
python3 scripts/02_hurst_distribution.py --universe sp100

# backtest the Tier-1 Hurst sort strategy
python3 scripts/03_hurst_sort_backtest.py --universe sp100

# compare fractional-differencing factor vs 12-1 momentum
python3 scripts/04_fracdiff_comparison.py
```

## Findings (2026-04-20, two passes done)

All net-of-cost Sharpes use 10bps per side with 95% bootstrap CI.

| Experiment                             | Stat       | 95% CI         | Verdict |
|----------------------------------------|-----------:|----------------|---------|
| S1: Hurst sort, S&P 100                | Sh ~0.23   | wide           | Weak    |
| S1: Hurst sort, S&P 600 (small caps)   | Sh -0.22   | [-0.78, 0.33]  | Null    |
| A3: FFD as standalone factor           | Sh  0.01   | wide           | Null    |
| S2: FFD + mom + vol in LightGBM        | Sh -0.08   | [-0.58, 0.43]  | Null    |
| S1: Hurst sort, ETFs (cross-asset)     | Sh -0.48   | [-1.06, 0.03]  | Negative (regime bet) |
| Residualized Hurst sort, ETFs          | Sh -0.13   | [-0.65, 0.38]  | Null    |
| **MF-DFA Δh VIX regime gate vs BH SPY**| **+0.10 Δ**| [-0.30, +0.51] | **Marginal (ns)** |
| Intraday execution: VWAP vs uniform    | -0.17 bps  | t = -1.0       | Null    |
| Baseline: 12-1 momentum                | Sh  0.14   | [-0.27, 0.64]  | Weak    |

Key findings:
1. **Narrow H dispersion in US equities.** Both S&P 100 and S&P 600 show
   mean H ~0.47, std ~0.03. Only ~5% reject Lo's null at 95% (chance).
2. **Leakage, not skill, drove GBM Sharpe 1.84.** 22-day embargo fix
   collapsed it to 0.015.
3. **Intraday volatility concentrates at market open** (2.4x average).
   Real microstructure finding. But "avoid open" execution strategy
   is *worse* than uniform on mean cost - high open vol cuts both ways.
4. **ETF Hurst sort is a long-commodities/short-bonds regime bet in
   fractal clothing.** Composition analysis revealed it.
5. **After controlling for asset class and vol, Hurst R^2 = 0.996 -
   nothing left to predict.** Residualized ETF sort: Sharpe -0.13,
   CI crosses zero.
6. **MF-DFA Delta h on VIX is the one signal pointing somewhere.** A
   regime gate ("flat when Delta h > median") improves buy-and-hold
   Sharpe from 0.59 to 0.70, but 95% bootstrap CI is [-0.30, +0.51] -
   not statistically significant.

**Interpretation.** Eight directional + execution tests. None reject
zero at 95%. The durable wins are the **infrastructure** (bootstrap CIs,
cost model, embargoed walk-forward, composition + residualization
audits) and three methodological lessons: audit embargo, audit
composition, residualize before claiming edge.

See `RESULTS.md` for full numbers and `IDEAS.md` for what's still worth doing.

## Caveats

- **Survivorship bias.** yfinance gives you currently-listed tickers only.
  Delistings are missing. All backtest Sharpes are biased upward.
- **No transaction costs in current backtests.** Applied only in the
  narrative assessment.
- **Single-run backtests, no bootstrap CI yet.** Statistical significance
  of any reported Sharpe is not established.
- Fractional-differencing test here is as a standalone factor, not the
  multi-feature ML use López de Prado proposed. That experiment is TBD.
