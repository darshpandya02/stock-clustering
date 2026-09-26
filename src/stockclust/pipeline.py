"""End-to-end refresh: download prices, build features, cluster, evaluate, write the snapshot JSON.

    uv run python -m stockclust.pipeline               # full refresh
    uv run python -m stockclust.pipeline --no-predict  # skip the walk-forward backtest
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import models
from .data import BENCHMARK, ROOT, fetch_prices, load_universe
from .features import FEATURE_LABELS, FEATURES, MIN_HISTORY, compute_features, standardize
from .prediction import HORIZON, build_panel, summarize, walk_forward
from .seeded_kmeans import kmeans as seeded_kmeans

SNAPSHOT = ROOT / "web" / "data" / "snapshot.json"


def _r(x: float, nd: int = 6):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd)


def build_snapshot(close: pd.DataFrame, universe: pd.DataFrame, predict: bool = True, seed: int = models.SEED) -> dict:
    bench = close[BENCHMARK]
    stocks = close.drop(columns=[BENCHMARK])

    history = stocks.notna().sum()
    feats = compute_features(stocks, bench).droplevel("date")
    as_of = stocks.index[-1]
    tickers = list(feats.index)
    excluded = sorted(set(universe.yahoo) - set(tickers))

    z_df, mean, std = standardize(feats)
    # Round once, and use the rounded values everywhere (sklearn, seeded
    # k-means and the browser) so all three see identical inputs.
    z = np.round(z_df.to_numpy(), 6)

    info = universe.set_index("yahoo").loc[tickers]
    sectors = info["sector"].to_numpy()

    sel = models.select_k(z, seed=seed)
    km = models.fit_kmeans(z, sel.best_k, seed=seed)
    km_labels = km.labels_
    km_sil = float(models.silhouette_or_none(z, km_labels))
    rand_mean, rand_p95 = models.random_label_ari(sectors, sel.best_k, seed=seed)

    db = models.fit_dbscan(z)
    db_labels = db.labels
    db_clusters = sorted(set(db_labels) - {-1})

    pca = models.fit_pca(z, 2, seed=seed)
    xy = pca.transform(z)

    s_labels, _, s_inertia = seeded_kmeans(z, sel.best_k, seed=seed)

    snapshot: dict = {
        "refreshed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "price_start": stocks.index[0].strftime("%Y-%m-%d"),
        "universe": {
            "name": "S&P 500 constituents",
            "n_constituents": int(len(universe)),
            "n_included": len(tickers),
            "excluded": [
                {
                    "ticker": t,
                    "trading_days": int(history.get(t, 0)),
                    "reason": "short history" if history.get(t, 0) < MIN_HISTORY else "no valid price on as-of date",
                }
                for t in excluded
            ],
            "min_history_days": MIN_HISTORY,
            "benchmark": BENCHMARK,
        },
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "stocks": [
            {
                "t": t,
                "name": info.loc[t, "name"],
                "sector": info.loc[t, "sector"],
                "industry": info.loc[t, "sub_industry"],
                "raw": [_r(v) for v in feats.loc[t, FEATURES]],
                "z": [float(v) for v in z[i]],
                "pca": [_r(xy[i, 0], 4), _r(xy[i, 1], 4)],
                "km": int(km_labels[i]),
                "db": int(db_labels[i]),
            }
            for i, t in enumerate(tickers)
        ],
        "pca": {
            "explained_variance_ratio": [_r(v, 4) for v in pca.explained_variance_ratio_],
            "loadings": [[_r(v, 4) for v in comp] for comp in pca.components_],
        },
        "kmeans": {
            "k": sel.best_k,
            "seed": seed,
            "n_init": 10,
            "silhouette": _r(km_sil, 4),
            "inertia": _r(km.inertia_, 2),
            "sizes": np.bincount(km_labels).tolist(),
            "sector_ari": _r(models.sector_ari(km_labels, sectors), 4),
            "random_ari_mean": _r(rand_mean, 4),
            "random_ari_p95": _r(rand_p95, 4),
            "selection": [
                {"k": int(r.k), "inertia": _r(r.inertia, 2), "silhouette": _r(r.silhouette, 4)}
                for r in sel.table.itertuples()
            ],
            "selection_rule": "highest silhouette for k >= 3",
            "browser_impl_ari_vs_sklearn": _r(models.ari(s_labels, km_labels), 4),
            "browser_impl_inertia": _r(s_inertia, 2),
        },
        "dbscan": {
            "eps": _r(db.eps, 4),
            "min_samples": db.min_samples,
            "n_clusters": len(db_clusters),
            "n_noise": int((db_labels == -1).sum()),
            "sizes": [int((db_labels == c).sum()) for c in db_clusters],
            "silhouette": _r(models.silhouette_or_none(z, db_labels), 4),
            "sector_ari": _r(models.sector_ari(db_labels, sectors), 4),
            "kdist": [_r(v, 4) for v in db.kdist],
            "knee_index": models.knee_index(db.kdist),
        },
    }

    if predict:
        t0 = time.time()
        panel = build_panel(stocks, bench)
        pred: dict = {
            "horizon_days": HORIZON,
            "k": sel.best_k,
            "initial_weeks": 52,
            "refit_every_weeks": 4,
            "panel_weeks": int(panel.index.get_level_values("date").nunique()),
            "panel_rows": int(len(panel)),
        }
        for target in ("y_raw", "y_excess"):
            preds = walk_forward(panel, target=target, k=sel.best_k, seed=seed)
            dates = preds.index.get_level_values("date")
            summary = summarize(preds)
            summary["test_start"] = dates.min().strftime("%Y-%m-%d")
            summary["test_end"] = dates.max().strftime("%Y-%m-%d")
            pred[target.removeprefix("y_")] = summary
        pred["seconds"] = round(time.time() - t0, 1)
        snapshot["prediction"] = pred
    return snapshot


def _round_tree(obj, nd: int = 4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_tree(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_tree(v, nd) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", type=Path, default=SNAPSHOT)
    args = ap.parse_args()

    universe = load_universe()
    close = fetch_prices(list(universe.yahoo) + [BENCHMARK], use_cache=not args.no_cache)
    if BENCHMARK not in close.columns:
        raise SystemExit("benchmark download failed; keeping the previous snapshot")
    snap = build_snapshot(close, universe, predict=not args.no_predict)
    if args.no_predict and args.out.exists():
        # Keep the last backtest so the site still shows it.
        prev = json.loads(args.out.read_text())
        if "prediction" in prev:
            snap["prediction"] = prev["prediction"]
    if "prediction" in snap:
        snap["prediction"] = _round_tree(snap["prediction"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snap, separators=(",", ":")))
    km, db = snap["kmeans"], snap["dbscan"]
    print(f"as_of={snap['as_of']} n={snap['universe']['n_included']}")
    print(f"kmeans k={km['k']} silhouette={km['silhouette']} sizes={km['sizes']} ARI_sector={km['sector_ari']}")
    print(f"dbscan eps={db['eps']} clusters={db['n_clusters']} noise={db['n_noise']} ARI_sector={db['sector_ari']}")
    if "prediction" in snap:
        for tgt in ("raw", "excess"):
            a = snap["prediction"][tgt]["accuracy"]
            print(tgt, {m: a[m]["accuracy"] for m in a})
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
