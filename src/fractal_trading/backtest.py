"""Cross-sectional sort backtest and performance stats.

Monthly rebalancing. Long top-quantile and short bottom-quantile of a factor.
Equal-weighted within each leg. Dollar-neutral.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series     # monthly long-short return
    long_returns: pd.Series
    short_returns: pd.Series
    holdings: pd.DataFrame           # dates x tickers, +1 long, -1 short, 0 flat
    turnover: float                  # average fraction of names changed per rebalance

    @property
    def stats(self) -> dict:
        r = self.portfolio_returns.dropna()
        if len(r) < 2:
            return {}
        ann_ret = r.mean() * 12
        ann_vol = r.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum = (1 + r).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        hit = (r > 0).mean()
        return {
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(dd),
            "hit_rate": float(hit),
            "n_periods": int(len(r)),
            "turnover": float(self.turnover),
        }


def cross_sectional_sort_backtest(
    factor: pd.DataFrame,          # dates x tickers (same index as prices)
    prices: pd.DataFrame,          # dates x tickers, daily close
    n_quantiles: int = 5,
    rebal_freq: str = "ME",        # month-end
    min_names_per_leg: int = 5,
) -> BacktestResult:
    """Sort tickers by factor at each rebal date; long top / short bottom.

    The factor at month-end M determines holdings for month M+1. No lookahead.
    """
    # align
    common_cols = factor.columns.intersection(prices.columns)
    factor = factor[common_cols].copy()
    prices = prices[common_cols].copy()

    rets = np.log(prices).diff()
    monthly_rets = rets.resample(rebal_freq).sum()          # log sum over the month
    factor_m = factor.resample(rebal_freq).last()           # factor at month-end

    # lag factor by one: factor at end of month M -> trades held during M+1
    signal = factor_m.shift(1)

    long_rets, short_rets, ls_rets = [], [], []
    holdings_list = []
    prev_long, prev_short = set(), set()
    turnover_list = []

    for date in signal.index:
        row = signal.loc[date].dropna()
        if len(row) < n_quantiles * min_names_per_leg:
            continue
        ranks = row.rank(pct=True)
        longs = ranks[ranks > (n_quantiles - 1) / n_quantiles].index
        shorts = ranks[ranks <= 1 / n_quantiles].index
        if len(longs) < min_names_per_leg or len(shorts) < min_names_per_leg:
            continue

        if date not in monthly_rets.index:
            continue
        period_ret = monthly_rets.loc[date]
        # convert log to simple for averaging, then back (small-diff approx OK)
        long_r = period_ret[longs].mean()
        short_r = period_ret[shorts].mean()
        long_rets.append((date, long_r))
        short_rets.append((date, short_r))
        ls_rets.append((date, long_r - short_r))

        h = pd.Series(0.0, index=common_cols)
        h.loc[longs] = 1.0 / len(longs)
        h.loc[shorts] = -1.0 / len(shorts)
        holdings_list.append(h.rename(date))

        cur_long, cur_short = set(longs), set(shorts)
        if prev_long or prev_short:
            turnover_list.append(
                len((cur_long ^ prev_long) | (cur_short ^ prev_short))
                / max(1, len(cur_long | cur_short | prev_long | prev_short))
            )
        prev_long, prev_short = cur_long, cur_short

    def _to_series(xs):
        if not xs:
            return pd.Series(dtype=float)
        idx, vals = zip(*xs)
        return pd.Series(vals, index=pd.DatetimeIndex(idx))

    holdings_df = pd.concat(holdings_list, axis=1).T if holdings_list else pd.DataFrame()
    return BacktestResult(
        portfolio_returns=_to_series(ls_rets),
        long_returns=_to_series(long_rets),
        short_returns=_to_series(short_rets),
        holdings=holdings_df,
        turnover=float(np.mean(turnover_list)) if turnover_list else np.nan,
    )


def rolling_factor(
    prices: pd.DataFrame,
    factor_fn,
    lookback_days: int,
    step_days: int = 21,
) -> pd.DataFrame:
    """Compute factor_fn(return_window) for each ticker on a rolling grid.

    Returns a DataFrame indexed by rebal-step dates with columns = tickers.
    """
    rets = np.log(prices).diff()
    n = len(rets)
    dates = rets.index[lookback_days::step_days]
    out = pd.DataFrame(index=dates, columns=prices.columns, dtype=float)

    for date in dates:
        end_loc = rets.index.get_loc(date)
        start_loc = end_loc - lookback_days
        if start_loc < 0:
            continue
        window = rets.iloc[start_loc:end_loc]
        for ticker in prices.columns:
            series = window[ticker].dropna().values
            if len(series) < lookback_days * 0.8:
                out.loc[date, ticker] = np.nan
                continue
            try:
                out.loc[date, ticker] = factor_fn(series)
            except Exception:
                out.loc[date, ticker] = np.nan
    return out.astype(float)
