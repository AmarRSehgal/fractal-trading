# fractal-trading

Personal research project: apply fractal geometry (Hurst exponent, DFA, fractional
differentiation, multifractal spectra) and Fourier / wavelet analysis to trading
and market analysis.

## Why

Most retail material on "fractal trading" (Elliott Wave, Bill Williams fractals)
is narrative, not math. This repo restricts itself to ideas that are (a) grounded
in peer-reviewed literature and (b) falsifiable with out-of-sample evaluation.

Equally, it rejects the most common misuse of Fourier in trading — running FFT on
raw price to find "cycles" — and instead uses spectral methods only where they
have a defensible signal (stationary features, seasonality, option pricing).

## Layout

- [`IDEAS.md`](IDEAS.md) - main deliverable. Concrete analysis and trading ideas
  with formulas, pseudocode, data needs, and evaluation plans.
- [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) - paper-by-paper notes and
  accessibility status (what was fetched vs. paywalled vs. book).
- `src/fractal_trading/` - library code (stubs only; to be filled in).
- `notebooks/` - exploratory work (to be added).

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data sources

Free sources to start:
- `yfinance` - daily equity/FX/crypto OHLCV
- `ccxt` - crypto spot/perp historical via exchange APIs
- Binance public data archive for minute bars

No paid feeds in this project.

## Status

Bootstrapped 2026-04-20. Read `IDEAS.md` first.
