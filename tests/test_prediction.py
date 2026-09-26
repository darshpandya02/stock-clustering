import numpy as np
import pandas as pd

from stockclust.prediction import build_panel, summarize, walk_forward


def test_panel_labels(prices):
    close, bench, _ = prices
    panel = build_panel(close, bench)
    d, t = panel.index[0]
    pos = close.index.get_loc(d)
    expected = close[t].iloc[pos + 5] / close[t].iloc[pos] - 1
    assert np.isclose(panel.loc[(d, t), "fwd_ret"], expected)
    assert panel.loc[(d, t), "y_raw"] == int(expected > 0)
    dates = panel.index.get_level_values("date").unique()
    gaps = np.diff([close.index.get_loc(x) for x in dates])
    assert (gaps == 5).all()


def test_walk_forward_uses_only_past(prices):
    """Scrambling labels from week w onward must not change predictions for weeks < w + refit."""
    close, bench, _ = prices
    panel = build_panel(close, bench)
    a = walk_forward(panel, k=3, initial_weeks=8, refit_every=4)
    weeks = sorted(panel.index.get_level_values("date").unique())
    cut = weeks[12]
    poisoned = panel.copy()
    later = poisoned.index.get_level_values("date") >= cut
    poisoned.loc[later, "y_raw"] = 1 - poisoned.loc[later, "y_raw"]
    b = walk_forward(poisoned, k=3, initial_weeks=8, refit_every=4)
    early = a.week < 13  # weeks 8..11 trained on < 8, weeks 12..15 trained on < 12
    cols = ["per_cluster_logit", "pooled_logit", "majority"]
    pd.testing.assert_frame_equal(a.loc[early, cols], b.loc[early, cols])


def test_walk_forward_deterministic_and_summary(prices):
    close, bench, _ = prices
    panel = build_panel(close, bench)
    a = walk_forward(panel, k=3, initial_weeks=8)
    b = walk_forward(panel, k=3, initial_weeks=8)
    pd.testing.assert_frame_equal(a, b)
    s = summarize(a)
    assert s["n_predictions"] == len(a)
    for m in ("per_cluster_logit", "pooled_logit", "majority", "persistence"):
        assert 0 <= s["accuracy"][m]["accuracy"] <= 1
