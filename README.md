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

| Experiment                             | Sharpe net | 95% CI         | Verdict |
|----------------------------------------|-----------:|----------------|---------|
| S1: Hurst sort, S&P 100                | ~0.23 (est)| wide           | Weak    |
| S1: Hurst sort, S&P 600 (small caps)   | -0.22      | [-0.78, 0.33]  | **Null** |
| A3: FFD as standalone factor           |  0.01      | wide           | **Null** |
| S2: FFD + mom + vol in LightGBM (embargoed) | -0.08 | [-0.58, 0.43] | **Null** |
| Baseline: 12-1 momentum                |  0.14      | [-0.27, 0.64]  | Weak    |
| S4: Intraday seasonality diagnostic    | N/A        | N/A            | **Real** (execution signal) |

Key findings:
1. **Narrow H dispersion.** Both S&P 100 and S&P 600 show mean H ~0.47 with
   std ~0.03. Only ~5% reject Lo's null at 95% (about what chance gives).
   My hypothesis that small-caps would have wider H was wrong.
2. **Leakage was the GBM Sharpe, not model skill.** First walk-forward
   gave Sharpe 1.84 - but training rows within 21 BDays of the test date
   had targets overlapping the test period. With a 22-day embargo the
   Sharpe collapsed to 0.015. See `scripts/07_gbm_walkforward.py`.
3. **Intraday volatility concentrates at market open** - 2.4x the mean
   of all intraday hours. Classical microstructure finding, but
   quantified per-stock and reusable.

**Interpretation.** Four directional fractal tests, four nulls. The
durable win of this project is the **infrastructure** (bootstrap CIs, TC
model, embargoed walk-forward). The intellectual win is the embargo
discovery - any future backtest in this repo must check embargo before
celebrating a Sharpe.

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
