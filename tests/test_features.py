import numpy as np
import pandas as pd
import pytest

from stockclust.features import FEATURES, MIN_HISTORY, _max_drawdown, compute_features, standardize


def test_shape_and_columns(prices):
    close, bench, _ = prices
    f = compute_features(close, bench)
    assert list(f.columns) == FEATURES
    assert len(f) == close.shape[1]
    assert np.isfinite(f.to_numpy()).all()


def test_returns_and_moving_averages(prices):
    close, bench, _ = prices
    f = compute_features(close, bench).droplevel("date")
    t = "T03"
    s = close[t]
    assert f.loc[t, "ret_21d"] == pytest.approx(s.iloc[-1] / s.iloc[-22] - 1)
    assert f.loc[t, "ret_252d"] == pytest.approx(s.iloc[-1] / s.iloc[-253] - 1)
    assert f.loc[t, "px_vs_sma50"] == pytest.approx(s.iloc[-1] / s.iloc[-50:].mean() - 1)
    assert f.loc[t, "sma50_vs_sma200"] == pytest.approx(s.iloc[-50:].mean() / s.iloc[-200:].mean() - 1)
    lr = np.log(s).diff().iloc[-21:]
    assert f.loc[t, "vol_21d"] == pytest.approx(lr.std() * np.sqrt(252))


def test_beta_matches_regression(prices):
    close, bench, _ = prices
    f = compute_features(close, bench).droplevel("date")
    r = close["T05"].pct_change().iloc[-252:]
    m = bench.pct_change().iloc[-252:]
    slope = np.polyfit(m, r, 1)[0]
    assert f.loc["T05", "beta_252d"] == pytest.approx(slope, rel=1e-9)


def test_beta_tracks_true_beta(prices):
    close, bench, betas = prices
    f = compute_features(close, bench).droplevel("date")
    corr = np.corrcoef(f["beta_252d"].to_numpy(), betas)[0, 1]
    assert corr > 0.8


def test_max_drawdown():
    w = np.array([[100.0], [120.0], [90.0], [110.0], [130.0]])
    assert _max_drawdown(w)[0] == pytest.approx(90 / 120 - 1)
    w_nan = np.array([[np.nan, 1.0], [1.0, 2.0]])
    out = _max_drawdown(w_nan)
    assert np.isnan(out[0]) and out[1] == 0.0


def test_no_lookahead(prices):
    """Features at date t must not change when later prices change."""
    close, bench, _ = prices
    d = close.index[300]
    base = compute_features(close, bench, [d])
    shocked = close.copy()
    shocked.iloc[301:] *= 3.0
    after = compute_features(shocked, bench, [d])
    pd.testing.assert_frame_equal(base, after)


def test_short_history_dropped(prices):
    close, bench, _ = prices
    close = close.copy()
    close.iloc[: len(close) - MIN_HISTORY + 10, 0] = np.nan
    f = compute_features(close, bench).droplevel("date")
    assert "T00" not in f.index
    assert len(f) == close.shape[1] - 1


def test_standardize(prices):
    close, bench, _ = prices
    f = compute_features(close, bench).droplevel("date")
    z, _, _ = standardize(f)
    assert np.allclose(z.mean(), 0, atol=1e-12)
    assert np.allclose(z.std(ddof=0), 1)
