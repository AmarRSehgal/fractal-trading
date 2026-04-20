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

## First-pass findings (2026-04-20)

On S&P 100, 2005-2026, survivorship-biased, no transaction costs:

| Experiment                             | Sharpe | Comment                      |
|----------------------------------------|--------|------------------------------|
| Hurst cross-sectional quintile sort    | 0.47   | ~0.2 after realistic costs   |
| FFD z-score as cross-sectional factor  | 0.01   | Underperformed 12-1 momentum |
| 12-1 momentum benchmark                | 0.17   | Weak on large-cap survivors  |

The Hurst-distribution diagnostic on S&P 100 reproduces Lo (1991): dispersion
is tight (5-95 percentile of H: [0.42, 0.52]); only 5/100 stocks reject the
no-long-memory null at 95% (about what random sampling gives).

**Interpretation:** retail fractal alpha on liquid US stocks is thin.
Next experiments should extend to Russell 2000 / small caps (expected wider
dispersion), add proper transaction costs and bootstrap CIs, and move frac-diff
from standalone factor into a multi-feature ML model (its intended use).

See `RESULTS.md` for the full numbers and `IDEAS.md` for what's next.

## Caveats

- **Survivorship bias.** yfinance gives you currently-listed tickers only.
  Delistings are missing. All backtest Sharpes are biased upward.
- **No transaction costs in current backtests.** Applied only in the
  narrative assessment.
- **Single-run backtests, no bootstrap CI yet.** Statistical significance
  of any reported Sharpe is not established.
- Fractional-differencing test here is as a standalone factor, not the
  multi-feature ML use López de Prado proposed. That experiment is TBD.
