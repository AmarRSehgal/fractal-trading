"""Stock universe helpers.

NOTE ON SURVIVORSHIP BIAS: yfinance-sourced universes contain currently-listed
tickers only. Delisted/bankrupt tickers are missing, which inflates backtest
returns in a way that cannot be fully corrected without a bias-free data
source (CRSP, Sharadar, etc.). All backtests in this repo are biased
upward. Treat Sharpe ratios with a skeptic's eye.
"""
from __future__ import annotations

import pandas as pd


SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_tickers() -> list[str]:
    """Current S&P 500 tickers via Wikipedia. Survivorship-biased."""
    tables = pd.read_html(SP500_WIKI)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return sorted(set(tickers))


def dow30_tickers() -> list[str]:
    """Dow Jones 30 - small, liquid universe for quick tests."""
    return sorted({
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
        "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
        "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V",
        "VZ", "WMT",
    })


def sp100_tickers() -> list[str]:
    """A static S&P 100 approximation - moderate universe size for backtests.
    Still survivorship-biased."""
    return sorted({
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT",
        "AMZN", "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY",
        "BRK-B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST",
        "CRM", "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DOW", "DUK",
        "EMR", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KO",
        "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET",
        "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE",
        "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX",
        "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
        "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC",
        "WMT", "XOM",
    })
