"""Clustering models: K-means++ with k chosen by silhouette, DBSCAN with eps from the k-distance knee, PCA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

SEED = 42


@dataclass
class KSelection:
    table: pd.DataFrame  # columns: k, inertia, silhouette
    best_k: int


def select_k(z: np.ndarray, k_values=range(2, 11), min_k: int = 3, seed: int = SEED) -> KSelection:
    """Fit K-means++ for each k and pick the k with the highest silhouette among k >= min_k.

    k = 2 usually scores highest on this data because it only splits off the
    most volatile names, which says little about structure, so it is reported
    but not selected.
    """
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(z)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(z, km.labels_))})
    table = pd.DataFrame(rows)
    eligible = table[table.k >= min_k]
    best_k = int(eligible.loc[eligible.silhouette.idxmax(), "k"])
    return KSelection(table=table, best_k=best_k)


def fit_kmeans(z: np.ndarray, k: int, seed: int = SEED) -> KMeans:
    return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(z)


def knee_index(sorted_values: np.ndarray) -> int:
    """Point of maximum distance below the chord joining the first and last points (Kneedle-style)."""
    y = np.asarray(sorted_values, dtype=float)
    n = len(y)
    if n < 3 or y[-1] == y[0]:
        return n - 1
    x = np.arange(n) / (n - 1)
    yn = (y - y[0]) / (y[-1] - y[0])
    return int(np.argmax(x - yn))


def k_distance(z: np.ndarray, min_samples: int) -> np.ndarray:
    """Sorted distance from each point to its min_samples-th neighbour (the point itself counts, as in DBSCAN)."""
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(z).kneighbors(z)
    return np.sort(dist[:, -1])


@dataclass
class DBSCANResult:
    labels: np.ndarray
    eps: float
    min_samples: int
    kdist: np.ndarray


def fit_dbscan(z: np.ndarray, min_samples: int | None = None) -> DBSCANResult:
    """DBSCAN with min_samples = 2 * n_features (Sander et al. heuristic) and eps at the k-distance knee."""
    if min_samples is None:
        min_samples = 2 * z.shape[1]
    kd = k_distance(z, min_samples)
    eps = float(kd[knee_index(kd)])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(z)
    return DBSCANResult(labels=labels, eps=eps, min_samples=min_samples, kdist=kd)


def silhouette_or_none(z: np.ndarray, labels: np.ndarray, ignore_noise: bool = True) -> float | None:
    labels = np.asarray(labels)
    mask = labels >= 0 if ignore_noise else np.ones(len(labels), bool)
    if len(set(labels[mask])) < 2:
        return None
    return float(silhouette_score(z[mask], labels[mask]))


def fit_pca(z: np.ndarray, n_components: int = 2, seed: int = SEED) -> PCA:
    return PCA(n_components=n_components, random_state=seed).fit(z)


def ari(a, b) -> float:
    return float(adjusted_rand_score(list(a), list(b)))


def sector_ari(labels, sectors) -> float:
    return ari(sectors, labels)


def random_label_ari(sectors, k: int, trials: int = 200, seed: int = SEED) -> tuple[float, float]:
    """ARI of random k-way labelings against sectors (mean, 95th percentile), as a reference point."""
    rng = np.random.default_rng(seed)
    sectors = list(sectors)
    vals = [adjusted_rand_score(sectors, rng.integers(0, k, len(sectors))) for _ in range(trials)]
    return float(np.mean(vals)), float(np.quantile(vals, 0.95))
