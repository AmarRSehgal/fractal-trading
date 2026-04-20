"""Cross-sectional sort backtest and performance stats.

Monthly rebalancing. Long top-quantile and short bottom-quantile of a factor.
Equal-weighted within each leg. Dollar-neutral. Optional transaction costs
and bootstrap confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series     # monthly long-short return (gross)
    long_returns: pd.Series
    short_returns: pd.Series
    holdings: pd.DataFrame           # dates x tickers, weights
    turnover: float                  # avg fraction of total weight changed per rebalance
    per_rebal_turnover: list[float] = field(default_factory=list)

    def net_returns(self, cost_bps_per_side: float) -> pd.Series:
        """Subtract transaction costs based on per-rebal turnover.

        cost_bps_per_side : one-way cost in bps. Round-trip = 2 * this.
        Turnover is per-rebal % of weight changed; we apply cost to the
        notional weight changed (both legs).
        """
        r = self.portfolio_returns.copy()
        if not self.per_rebal_turnover:
            return r
        # align: first return has no prior turnover (inception), then each
        # subsequent rebal pays cost based on turnover at that rebal
        # turnover_list[0] corresponds to second rebal
        cost = pd.Series(0.0, index=r.index)
        for i, to in enumerate(self.per_rebal_turnover):
            # to is fraction of (long+short combined) weight replaced
            # pay 2 * cost_bps_per_side (buying and selling the replaced piece)
            # on the fraction changed
            if i + 1 < len(cost):
                cost.iloc[i + 1] = to * 2 * cost_bps_per_side / 10_000
        return r - cost

    def stats(self, cost_bps_per_side: float = 0.0) -> dict:
        """Summary stats. Set cost_bps_per_side>0 to report net-of-cost."""
        r = self.portfolio_returns.dropna()
        net = self.net_returns(cost_bps_per_side).dropna() if cost_bps_per_side else r
        if len(net) < 2:
            return {}
        ann_ret = net.mean() * 12
        ann_vol = net.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum = (1 + net).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        hit = (net > 0).mean()
        return {
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(dd),
            "hit_rate": float(hit),
            "n_periods": int(len(net)),
            "turnover": float(self.turnover),
            "cost_bps_per_side": float(cost_bps_per_side),
        }

    def bootstrap_sharpe_ci(
        self,
        cost_bps_per_side: float = 0.0,
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> tuple[float, float, float]:
        """Bootstrap (low, point, high) confidence interval for annualized
        Sharpe of the monthly L/S returns. Uses i.i.d. resampling with
        replacement (assumes no serial dependence in the monthly L/S; usually
        an OK approximation for monthly rebalanced long-short portfolios).
        """
        r = (
            self.net_returns(cost_bps_per_side).dropna().values
            if cost_bps_per_side
            else self.portfolio_returns.dropna().values
        )
        if len(r) < 12:
            return (np.nan, np.nan, np.nan)
        rng = np.random.default_rng(seed)
        n = len(r)
        sharpes = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = r[idx]
            if sample.std() == 0:
                sharpes[b] = np.nan
            else:
                sharpes[b] = sample.mean() * 12 / (sample.std() * np.sqrt(12))
        sharpes = sharpes[np.isfinite(sharpes)]
        lo, hi = np.quantile(sharpes, [alpha / 2, 1 - alpha / 2])
        point = r.mean() * 12 / (r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
        return (float(lo), float(point), float(hi))

    def report(
        self,
        title: str = "backtest",
        cost_bps_per_side: float = 10.0,
        n_boot: int = 2000,
    ) -> str:
        """Pretty-printed report with gross and net stats plus bootstrap CI."""
        lines = [f"=== {title} ==="]
        gross = self.stats(cost_bps_per_side=0.0)
        net = self.stats(cost_bps_per_side=cost_bps_per_side)
        if not gross:
            return f"{title}: insufficient data"

        lines.append(f"{'metric':<20s} {'gross':>10s} {'net':>10s}")
        for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate"):
            lines.append(f"{k:<20s} {gross[k]:>10.4f} {net[k]:>10.4f}")
        lines.append(f"{'n_periods':<20s} {gross['n_periods']:>10d}")
        lines.append(f"{'turnover/rebal':<20s} {gross['turnover']:>10.4f}")
        lines.append(f"{'cost_bps_per_side':<20s} {cost_bps_per_side:>10.1f}")

        if n_boot > 0:
            lo_g, pt_g, hi_g = self.bootstrap_sharpe_ci(0.0, n_boot=n_boot)
            lo_n, pt_n, hi_n = self.bootstrap_sharpe_ci(cost_bps_per_side, n_boot=n_boot)
            lines.append(f"{'sharpe_ci_gross':<20s} [{lo_g:.3f}, {hi_g:.3f}]  pt {pt_g:.3f}")
            lines.append(f"{'sharpe_ci_net':<20s} [{lo_n:.3f}, {hi_n:.3f}]  pt {pt_n:.3f}")
        return "\n".join(lines)


def cross_sectional_sort_backtest(
    factor: pd.DataFrame,          # dates x tickers
    prices: pd.DataFrame,          # dates x tickers, daily close
    n_quantiles: int = 5,
    rebal_freq: str = "ME",        # month-end
    min_names_per_leg: int = 5,
) -> BacktestResult:
    """Sort tickers by factor at each rebal date; long top / short bottom.

    The factor at month-end M determines holdings for month M+1. No lookahead.
    """
    common_cols = factor.columns.intersection(prices.columns)
    factor = factor[common_cols].copy()
    prices = prices[common_cols].copy()

    rets = np.log(prices).diff()
    monthly_rets = rets.resample(rebal_freq).sum()
    factor_m = factor.resample(rebal_freq).last()
    signal = factor_m.shift(1)

    long_rets, short_rets, ls_rets = [], [], []
    holdings_list = []
    prev_h = pd.Series(0.0, index=common_cols)
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
        long_r = period_ret[longs].mean()
        short_r = period_ret[shorts].mean()
        long_rets.append((date, long_r))
        short_rets.append((date, short_r))
        ls_rets.append((date, long_r - short_r))

        h = pd.Series(0.0, index=common_cols)
        h.loc[longs] = 1.0 / len(longs)
        h.loc[shorts] = -1.0 / len(shorts)
        holdings_list.append(h.rename(date))

        # turnover as L1 distance in weights divided by 2 (gross weight changed)
        if prev_h.abs().sum() > 0:
            diff = (h - prev_h).abs().sum() / 2
            total_gross = h.abs().sum()
            turnover_list.append(float(diff / total_gross) if total_gross > 0 else np.nan)
        prev_h = h

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
        per_rebal_turnover=turnover_list,
    )


def rolling_factor(
    prices: pd.DataFrame,
    factor_fn,
    lookback_days: int,
    step_days: int = 21,
) -> pd.DataFrame:
    """Compute factor_fn(return_window) for each ticker on a rolling grid."""
    rets = np.log(prices).diff()
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
