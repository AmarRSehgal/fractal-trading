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
SP400_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"


def _from_wiki(url: str, symbol_col: str = "Symbol") -> list[str]:
    import io
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode()
    tables = pd.read_html(io.StringIO(html))
    for t in tables:
        if symbol_col in t.columns:
            tickers = t[symbol_col].astype(str).str.replace(".", "-", regex=False).tolist()
            return sorted(set(tickers))
    raise ValueError(f"No '{symbol_col}' column found in tables at {url}")


def sp500_tickers() -> list[str]:
    """Current S&P 500 tickers via Wikipedia. Survivorship-biased."""
    return _from_wiki(SP500_WIKI, "Symbol")


def sp400_tickers() -> list[str]:
    """Current S&P 400 (mid-cap) tickers. Survivorship-biased."""
    return _from_wiki(SP400_WIKI, "Symbol")


def sp600_tickers() -> list[str]:
    """Current S&P 600 (small-cap) tickers. Survivorship-biased."""
    return _from_wiki(SP600_WIKI, "Symbol")


def sp1500_tickers() -> list[str]:
    """Current S&P 500 + 400 + 600 = 1500 stocks. Survivorship-biased."""
    return sorted(set(sp500_tickers()) | set(sp400_tickers()) | set(sp600_tickers()))


def etf_universe() -> dict[str, str]:
    """Curated ETF universe with asset-class tags.

    Keys are yfinance tickers, values are asset-class labels. Selected for
    liquidity and availability of >= 15-year history. Intended for
    cross-asset cross-sectional studies where H dispersion may be wider
    than within pure US equities.
    """
    return {
        # US equity sectors (SPDR Select Sector)
        "XLK": "us_sector_tech", "XLF": "us_sector_fin", "XLE": "us_sector_energy",
        "XLV": "us_sector_health", "XLP": "us_sector_staples", "XLY": "us_sector_discr",
        "XLI": "us_sector_ind", "XLU": "us_sector_utility", "XLB": "us_sector_mat",
        # US broad / style
        "SPY": "us_broad", "QQQ": "us_broad", "IWM": "us_broad",
        "VTI": "us_broad", "IWB": "us_broad",
        "IWD": "us_style_lg_value", "IWF": "us_style_lg_growth",
        "IWN": "us_style_sm_value", "IWO": "us_style_sm_growth",
        # International developed
        "EFA": "intl_developed", "VEA": "intl_developed",
        "EWJ": "intl_japan", "EWG": "intl_germany", "EWU": "intl_uk",
        "EWA": "intl_australia", "EWC": "intl_canada", "EWH": "intl_hongkong",
        "EWI": "intl_italy", "EWP": "intl_spain", "EWQ": "intl_france",
        "EWL": "intl_switzerland", "EWT": "intl_taiwan", "EWS": "intl_singapore",
        "EWY": "intl_korea", "EZU": "intl_eurozone", "EPP": "intl_asiapac_exjp",
        # Emerging markets
        "EEM": "em_broad", "VWO": "em_broad",
        "INDA": "em_india", "FXI": "em_china",
        "EWZ": "em_brazil", "EWW": "em_mexico",
        "EPOL": "em_poland", "EZA": "em_safrica", "TUR": "em_turkey",
        # Fixed income
        "TLT": "bonds_lg_treasury", "IEF": "bonds_md_treasury",
        "SHY": "bonds_sh_treasury",
        "LQD": "bonds_ig_corp", "HYG": "bonds_hy_corp",
        "AGG": "bonds_aggregate", "TIP": "bonds_tips",
        "EMB": "bonds_em",
        # Commodities / FX
        "GLD": "commodity_gold", "SLV": "commodity_silver",
        "USO": "commodity_oil", "UNG": "commodity_natgas",
        "DBA": "commodity_ag", "DBC": "commodity_broad",
        "UUP": "fx_usd", "FXE": "fx_eur",
        # Real estate / alternatives
        "VNQ": "reit_us", "IYR": "reit_us", "PFF": "preferred_stock",
    }


def etf_tickers() -> list[str]:
    return sorted(etf_universe().keys())


def etf_asset_class(ticker: str) -> str:
    return etf_universe().get(ticker, "unknown")


def liquid_watchlist() -> list[str]:
    """Handful of very liquid US stocks / ETFs for intraday studies."""
    return sorted({
        "SPY", "QQQ", "IWM", "DIA",
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "AMD", "AVGO", "NFLX", "INTC",
        "JPM", "BAC", "GS",
        "XOM", "CVX",
        "WMT", "COST", "HD",
        "JNJ", "UNH", "LLY",
    })


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
