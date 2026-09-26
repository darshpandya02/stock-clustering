import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    """Synthetic geometric random walks for 30 tickers plus a benchmark, 400 trading days."""
    rng = np.random.default_rng(0)
    n_days, n_tickers = 400, 30
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    market = rng.normal(0.0004, 0.01, n_days)
    betas = rng.uniform(0.3, 1.8, n_tickers)
    rets = market[:, None] * betas + rng.normal(0, 0.015, (n_days, n_tickers))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"T{i:02d}" for i in range(n_tickers)])
    bench = pd.Series(100 * np.exp(np.cumsum(market)), index=idx, name="SPY")
    return close, bench, betas
