"""Price-derived features for each stock, computed as of one or more dates.

Every feature at date t uses only prices up to and including t, so the same
code builds both the latest snapshot and the historical panel used for the
walk-forward movement prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

FEATURES: list[str] = [
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "ret_252d",
    "roc_accel_21d",
    "vol_21d",
    "vol_63d",
    "px_vs_sma20",
    "px_vs_sma50",
    "px_vs_sma200",
    "sma50_vs_sma200",
    "beta_252d",
    "max_drawdown_252d",
]

FEATURE_LABELS: dict[str, str] = {
    "ret_5d": "1-week return",
    "ret_21d": "1-month return",
    "ret_63d": "3-month return",
    "ret_126d": "6-month return",
    "ret_252d": "12-month return",
    "roc_accel_21d": "Change in 1-month return vs prior month",
    "vol_21d": "1-month volatility (annualized)",
    "vol_63d": "3-month volatility (annualized)",
    "px_vs_sma20": "Price vs 20-day moving average",
    "px_vs_sma50": "Price vs 50-day moving average",
    "px_vs_sma200": "Price vs 200-day moving average",
    "sma50_vs_sma200": "50-day vs 200-day moving average",
    "beta_252d": "Beta vs SPY (1 year)",
    "max_drawdown_252d": "Max drawdown (1 year)",
}

# Rows of history needed before the first valid feature value.
MIN_HISTORY = TRADING_DAYS + 1


def _max_drawdown(window: np.ndarray) -> np.ndarray:
    """Max drawdown per column of a (days x tickers) price window, as a negative fraction."""
    running_max = np.fmax.accumulate(window, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = window / running_max - 1.0
    incomplete = np.isnan(window).any(axis=0)
    out = np.full(window.shape[1], np.nan)
    if (~incomplete).any():
        out[~incomplete] = dd[:, ~incomplete].min(axis=0)
    return out


def compute_features(
    close: pd.DataFrame, benchmark: pd.Series, dates: list[pd.Timestamp] | None = None
) -> pd.DataFrame:
    """Feature table indexed by (date, ticker).

    close: adjusted closes, index = trading dates, columns = tickers.
    benchmark: adjusted closes of the benchmark (SPY) on the same index.
    dates: dates to evaluate at; defaults to the last row only.
    """
    close = close.sort_index()
    benchmark = benchmark.reindex(close.index)
    if dates is None:
        dates = [close.index[-1]]
    dates = [pd.Timestamp(d) for d in dates]

    daily = close.pct_change(fill_method=None)
    bench_ret = benchmark.pct_change(fill_method=None)
    log_ret = np.log(close).diff()

    frames: dict[str, pd.DataFrame] = {}
    for n in (5, 21, 63, 126, 252):
        frames[f"ret_{n}d" if n != 252 else "ret_252d"] = close / close.shift(n) - 1.0
    frames["roc_accel_21d"] = frames["ret_21d"] - frames["ret_21d"].shift(21)
    frames["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    frames["vol_63d"] = log_ret.rolling(63).std() * np.sqrt(TRADING_DAYS)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    frames["px_vs_sma20"] = close / sma20 - 1.0
    frames["px_vs_sma50"] = close / sma50 - 1.0
    frames["px_vs_sma200"] = close / sma200 - 1.0
    frames["sma50_vs_sma200"] = sma50 / sma200 - 1.0

    # Rolling beta = cov(r, r_m) / var(r_m) over 252 days.
    w = TRADING_DAYS
    rm = bench_ret
    mean_r = daily.rolling(w).mean()
    mean_m = rm.rolling(w).mean()
    mean_rm = daily.mul(rm, axis=0).rolling(w).mean()
    var_m = rm.rolling(w).var(ddof=0)
    cov = mean_rm.sub(mean_r.mul(mean_m, axis=0))
    frames["beta_252d"] = cov.div(var_m, axis=0)

    idx = close.index
    values = close.to_numpy(dtype=float)
    rows = []
    for d in dates:
        pos = idx.get_loc(d)
        if pos < MIN_HISTORY - 1:
            continue
        row = pd.DataFrame({name: frames[name].iloc[pos] for name in FEATURES if name != "max_drawdown_252d"})
        row["max_drawdown_252d"] = _max_drawdown(values[pos - w + 1 : pos + 1])
        row = row[FEATURES]
        row.index.name = "ticker"
        row["date"] = d
        rows.append(row.reset_index())
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", *FEATURES]).set_index(["date", "ticker"])
    out = pd.concat(rows, ignore_index=True).set_index(["date", "ticker"])
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each column to its cross-sectional [lower, upper] quantiles."""
    lo = df.quantile(lower)
    hi = df.quantile(upper)
    return df.clip(lower=lo, upper=hi, axis=1)


def standardize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Winsorize, then z-score each column. Returns (z, mean, std) of the winsorized data."""
    w = winsorize(df)
    mean = w.mean()
    std = w.std(ddof=0).replace(0, 1.0)
    return (w - mean) / std, mean, std
