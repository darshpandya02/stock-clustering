"""Next-week direction prediction per cluster, evaluated walk-forward.

Setup
- Samples: every stock at every 5th trading day (weekly), with the same
  features as the clustering, z-scored cross-sectionally within each week.
- Target: whether the stock's next-5-trading-day return is positive
  ("raw"), and separately whether it beats SPY over that week ("excess").
- Walk-forward: models are refit every `refit_every` weeks on all earlier
  weeks only (expanding window). A week's label is known by the next week's
  feature date, so training on weeks < t to predict week t uses no future data.
- Clusters: K-means is refit on the training rows at each refit and test rows
  are assigned to the nearest centroid, so cluster membership is not taken
  from the future either. One logistic regression is fit per cluster.
- Baselines: majority class of the training labels, and "same direction as
  the past week".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from .features import FEATURES, MIN_HISTORY, compute_features, winsorize

HORIZON = 5


def build_panel(close: pd.DataFrame, benchmark: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    close = close.sort_index()
    benchmark = benchmark.reindex(close.index)
    last = len(close.index) - 1
    positions = list(range(last - horizon, MIN_HISTORY - 2, -horizon))[::-1]
    dates = [close.index[p] for p in positions]
    feats = compute_features(close, benchmark, dates)

    fwd = close.shift(-horizon) / close - 1.0
    bench_fwd = benchmark.shift(-horizon) / benchmark - 1.0
    fwd_long = fwd.loc[dates].stack().rename("fwd_ret")
    fwd_long.index.names = ["date", "ticker"]
    panel = feats.join(fwd_long, how="inner")
    panel["bench_fwd_ret"] = bench_fwd.reindex(panel.index.get_level_values("date")).to_numpy()
    panel = panel.dropna()
    panel["y_raw"] = (panel["fwd_ret"] > 0).astype(int)
    panel["y_excess"] = (panel["fwd_ret"] > panel["bench_fwd_ret"]).astype(int)

    # Cross-sectional z-score within each week.
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        w = winsorize(g[FEATURES])
        sd = w.std(ddof=0).replace(0, 1.0)
        return (w - w.mean()) / sd

    z = panel.groupby(level="date", group_keys=False)[FEATURES].apply(_z)
    z.columns = [f"z_{c}" for c in FEATURES]
    return panel.join(z)


def walk_forward(
    panel: pd.DataFrame,
    target: str = "y_raw",
    k: int = 4,
    initial_weeks: int = 52,
    refit_every: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Out-of-sample predictions for every week after the initial training window."""
    zcols = [f"z_{c}" for c in FEATURES]
    weeks = np.array(sorted(panel.index.get_level_values("date").unique()))
    week_idx = pd.Series(np.arange(len(weeks)), index=weeks)
    wk = week_idx.reindex(panel.index.get_level_values("date")).to_numpy()
    X = panel[zcols].to_numpy()
    y = panel[target].to_numpy()
    past_up = (panel["ret_5d"] > 0).astype(int).to_numpy()
    if target == "y_excess":
        # Persistence baseline for the excess target: did it beat SPY last week?
        # Approximated by last week's return being above that week's median.
        med = panel.groupby(level="date")["ret_5d"].transform("median").to_numpy()
        past_up = (panel["ret_5d"].to_numpy() > med).astype(int)

    out = []
    for start in range(initial_weeks, len(weeks), refit_every):
        train = wk < start
        test = (wk >= start) & (wk < start + refit_every)
        if not test.any():
            continue
        Xtr, ytr = X[train], y[train]
        km = KMeans(n_clusters=k, init="k-means++", n_init=4, random_state=seed).fit(Xtr)
        ctr = km.labels_
        cte = km.predict(X[test])
        majority = int(ytr.mean() >= 0.5)

        pooled = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        pred_pooled = pooled.predict(X[test])

        pred_cluster = np.empty(test.sum(), dtype=int)
        pred_majority = np.full(test.sum(), majority)
        for c in range(k):
            m_tr = ctr == c
            m_te = cte == c
            if not m_te.any():
                continue
            yc = ytr[m_tr]
            if len(np.unique(yc)) < 2:
                pred_cluster[m_te] = int(yc.mean() >= 0.5) if len(yc) else majority
                continue
            clf = LogisticRegression(max_iter=1000).fit(Xtr[m_tr], yc)
            pred_cluster[m_te] = clf.predict(X[test][m_te])

        out.append(
            pd.DataFrame(
                {
                    "week": wk[test],
                    "cluster": cte,
                    "y": y[test],
                    "per_cluster_logit": pred_cluster,
                    "pooled_logit": pred_pooled,
                    "majority": pred_majority,
                    "persistence": past_up[test],
                },
                index=panel.index[test],
            )
        )
    return pd.concat(out)


def summarize(preds: pd.DataFrame) -> dict:
    """Accuracy per method, with a week-clustered standard error.

    Stocks in the same week move together, so the rows are not independent;
    the uncertainty is computed from per-week accuracies instead.
    """
    methods = ["per_cluster_logit", "pooled_logit", "majority", "persistence"]
    weekly = preds.groupby("week").apply(
        lambda g: pd.Series({m: float((g[m] == g["y"]).mean()) for m in methods}), include_groups=False
    )
    n_weeks = len(weekly)
    res: dict = {
        "n_predictions": int(len(preds)),
        "n_weeks": int(n_weeks),
        "up_rate": float(preds["y"].mean()),
        "accuracy": {},
    }
    for m in methods:
        acc = float((preds[m] == preds["y"]).mean())
        se = float(weekly[m].std(ddof=1) / np.sqrt(n_weeks))
        res["accuracy"][m] = {"accuracy": acc, "weekly_se": se}
    diff = weekly["per_cluster_logit"] - weekly["majority"]
    res["per_cluster_minus_majority"] = {
        "mean": float(diff.mean()),
        "weekly_se": float(diff.std(ddof=1) / np.sqrt(n_weeks)),
        "weeks_model_better": int((diff > 0).sum()),
        "weeks_model_worse": int((diff < 0).sum()),
    }
    return res
