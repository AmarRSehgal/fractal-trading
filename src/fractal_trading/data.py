"""yfinance-backed price data with local parquet cache.

Downloads adjusted-close prices for a ticker list and caches them to
.data_cache/ under the repo root. Re-running uses the cache.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".data_cache"


def cache_path(tickers: list[str], start: str, end: str, interval: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    key = f"{'_'.join(sorted(tickers))[:80]}_{start}_{end}_{interval}"
    # hash long keys to avoid filesystem path length issues
    if len(key) > 100:
        import hashlib
        key = hashlib.sha256("_".join(sorted(tickers)).encode()).hexdigest()[:16]
        key = f"{key}_{start}_{end}_{interval}_n{len(tickers)}"
    return CACHE_DIR / f"{key}.parquet"


def load_prices(
    tickers: list[str],
    start: str = "2005-01-01",
    end: str | None = None,
    interval: str = "1d",
    field: str = "Close",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download (or load from cache) a wide-format price DataFrame.

    Rows: dates. Columns: tickers. Values: adjusted close by default.
    Auto-adjusted, so dividends/splits are reflected in Close.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    path = cache_path(tickers, start, end, interval)
    if use_cache and path.exists():
        df = pd.read_parquet(path)
        return df

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            df = raw[field]
        else:
            df = raw.xs(field, axis=1, level=-1)
    else:
        df = raw[[field]].rename(columns={field: tickers[0]})

    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if use_cache:
        df.to_parquet(path)
    return df


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    return np.log(prices).diff()
