"""Execution cost simulation: VWAP-profile vs uniform vs naive schedules.

Scenario: each trading day we need to buy a fixed notional in a single
stock, spread across the trading day's hourly bars. Compare different
allocation schedules and measure execution shortfall vs the day's VWAP.

Strategies:
  - uniform      : equal weight each hour
  - vwap_profile : allocate proportional to each stock's learned average
                   volume per hour-of-day (fitted on prior data only)
  - avoid_open   : skip the first bar (open is 2.4x vol - Scripts 06)
  - front_load   : everything in first hour
  - end_load     : everything in last hour

Metric: implementation shortfall vs day's VWAP, in basis points.

Uses hourly bars for 26 watchlist stocks over the last ~700 days. Profile
fitted on first 350 days; test on last 350 days.

Usage:
    python3 scripts/11_execution_backtest.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yfinance as yf

from fractal_trading.universe import liquid_watchlist


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def load_hourly_ohlcv(tickers: list, days: int = 700) -> pd.DataFrame:
    """Return multi-level DataFrame (hour_ts, ticker) -> (Open, High, Low, Close, Volume)."""
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    df = yf.download(
        tickers, start=start, end=end, interval="1h",
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    return df


def fit_volume_profile(panel: pd.DataFrame) -> pd.Series:
    """Given an hourly panel for ONE ticker (datetime index, OHLCV columns),
    return a Series indexed by hour-of-day giving normalized avg volume."""
    df = panel.copy()
    df["hour"] = df.index.hour
    by_hour = df.groupby("hour")["Volume"].mean()
    if by_hour.sum() <= 0:
        return pd.Series(dtype=float)
    return by_hour / by_hour.sum()


def day_vwap(day_df: pd.DataFrame) -> float:
    """VWAP of a day's hourly bars using (H+L+C)/3 * Volume."""
    typical = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3
    total_vol = day_df["Volume"].sum()
    if total_vol <= 0:
        return float(day_df["Close"].mean())
    return float((typical * day_df["Volume"]).sum() / total_vol)


def strategy_costs(day_df: pd.DataFrame, profile: pd.Series) -> dict:
    """For one day's bars, simulate each strategy and return avg fill price
    for each strategy, plus the day's VWAP as benchmark."""
    if len(day_df) < 3:
        return {}
    df = day_df.sort_index()
    df = df.copy()
    df["hour"] = df.index.hour
    n = len(df)
    close = df["Close"].values

    vwap = day_vwap(df)

    # 1) uniform
    uniform_w = np.ones(n) / n
    # 2) vwap_profile from learned per-hour volume shares
    hrs = df["hour"].values
    vp_w = np.array([profile.get(h, 1.0 / n) for h in hrs])
    # renormalize (hours in this day's bars may be subset of profile)
    if vp_w.sum() == 0:
        vp_w = uniform_w.copy()
    else:
        vp_w = vp_w / vp_w.sum()
    # 3) avoid open: skip first bar
    avoid_open_w = np.ones(n) / max(n - 1, 1)
    avoid_open_w[0] = 0.0
    if avoid_open_w.sum() > 0:
        avoid_open_w = avoid_open_w / avoid_open_w.sum()
    # 4) front-load: all weight on first bar
    front_w = np.zeros(n); front_w[0] = 1.0
    # 5) end-load: all weight on last bar
    end_w = np.zeros(n); end_w[-1] = 1.0

    fills = {
        "uniform": float((uniform_w * close).sum()),
        "vwap_profile": float((vp_w * close).sum()),
        "avoid_open": float((avoid_open_w * close).sum()),
        "front_load": float((front_w * close).sum()),
        "end_load": float((end_w * close).sum()),
    }
    shortfall_bps = {
        name: 1e4 * (fill - vwap) / vwap for name, fill in fills.items()
    }
    shortfall_bps["vwap"] = 0.0  # benchmark
    return shortfall_bps


def main():
    tickers = liquid_watchlist()
    print(f"Loading hourly bars for {len(tickers)} tickers, last 700 days...")
    raw = load_hourly_ohlcv(tickers, days=700)

    if isinstance(raw.columns, pd.MultiIndex):
        tickers_ok = [t for t in tickers if (t, "Close") in raw.columns]
    else:
        tickers_ok = tickers
    print(f"  tickers with data: {len(tickers_ok)}")

    all_results = []
    per_ticker_shortfall = {}

    for tkr in tickers_ok:
        if isinstance(raw.columns, pd.MultiIndex):
            panel = raw[tkr].dropna(subset=["Close"])
        else:
            panel = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        if len(panel) < 200:
            continue

        # split: first half fit profile, second half test
        midpoint = len(panel) // 2
        fit_panel = panel.iloc[:midpoint]
        test_panel = panel.iloc[midpoint:]
        profile = fit_volume_profile(fit_panel)
        if profile.empty:
            continue

        # iterate through trading days in test
        test_panel = test_panel.copy()
        test_panel["date"] = test_panel.index.date
        shortfalls = []
        for date, day_df in test_panel.groupby("date"):
            if len(day_df) < 4:
                continue
            cost = strategy_costs(day_df, profile)
            if cost:
                cost["date"] = date; cost["ticker"] = tkr
                shortfalls.append(cost)
        if shortfalls:
            df_st = pd.DataFrame(shortfalls)
            per_ticker_shortfall[tkr] = df_st
            all_results.append(df_st)

    if not all_results:
        print("No results.")
        return
    big = pd.concat(all_results, ignore_index=True)
    big.to_csv(RESULTS / "execution_shortfall_bps.csv", index=False)

    # Summary: per-strategy mean and median shortfall across all (day, ticker) pairs
    strat_cols = ["uniform", "vwap_profile", "avoid_open", "front_load", "end_load", "vwap"]
    print()
    print("=" * 72)
    print("Implementation shortfall vs day VWAP, all (day x ticker), bps")
    print(f"N observations: {len(big):,}")
    print("-" * 72)
    summary = pd.DataFrame({
        "mean_bps": big[strat_cols].mean(),
        "median_bps": big[strat_cols].median(),
        "std_bps": big[strat_cols].std(),
        "p10_bps": big[strat_cols].quantile(0.10),
        "p90_bps": big[strat_cols].quantile(0.90),
    })
    print(summary.round(2).to_string())
    summary.to_csv(RESULTS / "execution_shortfall_summary.csv")

    # Pairwise test: does vwap_profile beat uniform consistently?
    diff = big["uniform"] - big["vwap_profile"]  # positive = uniform was more expensive = vp better
    mean_diff = diff.mean()
    t_stat = mean_diff / (diff.std() / np.sqrt(len(diff)))
    print()
    print(f"VWAP-profile vs uniform: mean diff = {mean_diff:+.2f} bps, t-stat = {t_stat:.2f}")
    diff_ao = big["uniform"] - big["avoid_open"]
    print(f"avoid-open  vs uniform: mean diff = {diff_ao.mean():+.2f} bps, t-stat = "
          f"{diff_ao.mean() / (diff_ao.std() / np.sqrt(len(diff_ao))):.2f}")

    # Per-ticker: which stocks benefit most from avoid-open?
    per_tkr_mean = big.groupby("ticker")[["uniform", "avoid_open", "vwap_profile"]].mean()
    per_tkr_mean["avoid_open_save"] = per_tkr_mean["uniform"] - per_tkr_mean["avoid_open"]
    print("\nTop 5 tickers where avoid-open saves the most:")
    print(per_tkr_mean.nlargest(5, "avoid_open_save")[["uniform", "avoid_open", "avoid_open_save"]].round(2))
    print("\nTop 5 where avoid-open HURTS most:")
    print(per_tkr_mean.nsmallest(5, "avoid_open_save")[["uniform", "avoid_open", "avoid_open_save"]].round(2))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        melt = big[strat_cols].melt(var_name="strategy", value_name="shortfall_bps")
        # Boxplot
        strats = ["front_load", "uniform", "vwap_profile", "avoid_open", "end_load"]
        data = [big[s].values for s in strats]
        ax.boxplot(data, labels=strats, showfliers=False)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="VWAP benchmark")
        ax.set_ylabel("Implementation shortfall vs VWAP (bps)")
        ax.set_title(f"Per-day per-stock execution shortfall (n={len(big):,})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "execution_shortfall.png", dpi=120)
        print(f"\nSaved: {RESULTS / 'execution_shortfall.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
