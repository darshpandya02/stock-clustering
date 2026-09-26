"""Price download from Yahoo Finance (via yfinance) with a local parquet cache."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONSTITUENTS_CSV = ROOT / "data" / "sp500_constituents.csv"
CACHE_DIR = ROOT / "data" / "cache"
BENCHMARK = "SPY"


def load_universe() -> pd.DataFrame:
    """S&P 500 constituents with GICS sector labels, keyed by Yahoo symbol."""
    df = pd.read_csv(CONSTITUENTS_CSV)
    df["yahoo"] = df["symbol"].str.replace(".", "-", regex=False)
    return df


def _download_batch(tickers: list[str], period: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if raw is None or raw.empty:
                raise RuntimeError("empty response")
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            return close
        except Exception as err:  # network errors and rate limits
            last_err = err
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed for batch starting {tickers[0]}: {last_err}")


def fetch_prices(
    tickers: list[str],
    period: str = "5y",
    batch_size: int = 50,
    pause: float = 2.0,
    max_age_hours: float = 12.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker, index = trading dates.

    Downloads in small batches with a pause between them to stay under Yahoo's
    rate limits, and reuses a cached parquet file when it is fresh enough.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"close_{period}.parquet"
    if use_cache and cache.exists():
        age_h = (time.time() - cache.stat().st_mtime) / 3600
        cached = pd.read_parquet(cache)
        if age_h < max_age_hours and set(tickers) <= set(cached.columns):
            return cached[tickers]

    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        frames.append(_download_batch(batch, period))
        if i + batch_size < len(tickers):
            time.sleep(pause)
    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    close.to_parquet(cache)
    return close[[t for t in tickers if t in close.columns]]
