"""S4: intraday seasonality from hourly bars. Fourier / average-by-bucket
analysis to identify when liquidity concentrates within the trading day.

yfinance gives 1-hour bars for up to 730 days. We compute:
  - Average |log return| by hour-of-day (volatility seasonality)
  - Average volume by hour-of-day
  - Welch PSD on |log return| to confirm 6-hour and daily peaks

The actionable output is a per-stock 'best hours to trade' ranking.
Not a directional strategy on its own - execution alpha.

Usage:
    python3 scripts/06_intraday_seasonality.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.signal import welch

from fractal_trading.data import load_prices
from fractal_trading.universe import liquid_watchlist


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def hour_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour + idx.minute / 60.0


def seasonality_by_hour(ohlcv_close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Given hourly close and volume, return avg |ret| and avg volume per
    hour-of-day bucket."""
    ret = np.log(ohlcv_close).diff()
    abs_ret = ret.abs()
    # only trading hours (roughly 09:30 to 16:00 ET -> 13:30 to 20:00 UTC)
    df = pd.DataFrame({"abs_ret": abs_ret, "volume": volume, "hour": ohlcv_close.index.hour})
    bucket = df.groupby("hour").agg({"abs_ret": "mean", "volume": "mean"})
    bucket["n"] = df.groupby("hour").size()
    return bucket


def main():
    tickers = liquid_watchlist()
    print(f"Loading 1-hour bars for {len(tickers)} watchlist tickers (max 730 days)...")

    # yfinance 1h interval limitation: ~730 days
    end = pd.Timestamp.today()
    start = (end - pd.Timedelta(days=700)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    close_prices = load_prices(tickers, start=start, end=end_str, interval="1h", field="Close")
    volumes = load_prices(tickers, start=start, end=end_str, interval="1h", field="Volume")
    print(f"  close shape {close_prices.shape}, volume shape {volumes.shape}")
    print(f"  date range: {close_prices.index.min()} to {close_prices.index.max()}")

    # Per-ticker seasonality
    records = []
    spec_records = []
    for tkr in tickers:
        if tkr not in close_prices.columns:
            continue
        cp = close_prices[tkr].dropna()
        vv = volumes[tkr].reindex(cp.index).fillna(0)
        if len(cp) < 200:
            continue
        b = seasonality_by_hour(cp, vv)
        # normalize by within-stock mean for comparability
        b["abs_ret_norm"] = b["abs_ret"] / b["abs_ret"].mean()
        b["volume_norm"] = b["volume"] / b["volume"].mean()
        for hr, row in b.iterrows():
            records.append({
                "ticker": tkr, "hour_utc": int(hr),
                "abs_ret": row["abs_ret"],
                "volume": row["volume"],
                "abs_ret_norm": row["abs_ret_norm"],
                "volume_norm": row["volume_norm"],
                "n_bars": int(row["n"]),
            })

        # Welch PSD on |log returns|
        ret = np.log(cp).diff().dropna().values
        abs_ret = np.abs(ret)
        if len(abs_ret) > 128:
            freqs, psd = welch(abs_ret, fs=1.0, nperseg=min(256, len(abs_ret) // 2))
            for f, p in zip(freqs, psd):
                spec_records.append({"ticker": tkr, "freq_per_bar": float(f), "psd": float(p)})

    df = pd.DataFrame(records)
    spec_df = pd.DataFrame(spec_records)
    if df.empty:
        print("No data collected.")
        return

    # Aggregate across universe
    agg = df.groupby("hour_utc").agg(
        abs_ret_norm=("abs_ret_norm", "mean"),
        volume_norm=("volume_norm", "mean"),
        n_stocks=("ticker", "nunique"),
    ).reset_index()
    print()
    print("Average |return| and volume by hour-of-day (UTC), normalized to per-stock mean:")
    print("-" * 72)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Identify best (highest volume) and worst hours across universe
    best_hours = agg.sort_values("volume_norm", ascending=False).head(3)["hour_utc"].tolist()
    worst_hours = agg.sort_values("volume_norm").head(3)["hour_utc"].tolist()
    print(f"\nTop 3 highest-volume hours (UTC): {best_hours}")
    print(f"Bottom 3 lowest-volume hours (UTC): {worst_hours}")

    df.to_csv(RESULTS / "intraday_seasonality.csv", index=False)
    agg.to_csv(RESULTS / "intraday_seasonality_agg.csv", index=False)
    spec_df.to_csv(RESULTS / "intraday_psd.csv", index=False)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        axes[0, 0].bar(agg["hour_utc"], agg["volume_norm"])
        axes[0, 0].set_title("Avg volume by hour-of-day (normalized)")
        axes[0, 0].set_xlabel("Hour UTC"); axes[0, 0].set_ylabel("Volume / per-stock mean")
        axes[0, 0].axhline(1.0, color="red", linestyle="--", alpha=0.5)
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].bar(agg["hour_utc"], agg["abs_ret_norm"])
        axes[0, 1].set_title("Avg |log return| by hour-of-day (normalized)")
        axes[0, 1].set_xlabel("Hour UTC"); axes[0, 1].set_ylabel("|Return| / per-stock mean")
        axes[0, 1].axhline(1.0, color="red", linestyle="--", alpha=0.5)
        axes[0, 1].grid(alpha=0.3)

        # Average PSD across stocks
        if not spec_df.empty:
            psd_avg = spec_df.groupby("freq_per_bar")["psd"].mean().reset_index()
            axes[1, 0].loglog(psd_avg["freq_per_bar"], psd_avg["psd"])
            axes[1, 0].set_title("Mean PSD of |log returns| (hourly bars, log-log)")
            axes[1, 0].set_xlabel("Frequency (cycles per bar)"); axes[1, 0].set_ylabel("PSD")
            axes[1, 0].axvline(1/7, color="red", linestyle="--", alpha=0.5, label="~7 bar (daily)")
            axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3, which="both")

        axes[1, 1].axis("off")
        axes[1, 1].text(0.0, 0.9,
            "Notes on hour-UTC mapping:\n"
            "  US equities trade 09:30-16:00 ET.\n"
            "  In winter EST (UTC-5): 14:30-21:00 UTC.\n"
            "  In summer EDT (UTC-4): 13:30-20:00 UTC.\n"
            "  yfinance 1h bars are stamped at bar START.\n"
            "\n"
            f"Top volume hours (UTC): {best_hours}\n"
            f"These are your execution-friendly windows.",
            fontsize=10, family="monospace", va="top")
        fig.tight_layout()
        png = RESULTS / "intraday_seasonality.png"
        fig.savefig(png, dpi=120)
        print(f"Saved: {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
