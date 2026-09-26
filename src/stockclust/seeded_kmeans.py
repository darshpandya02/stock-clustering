"""K-means++ with a portable PRNG, written to match web/kmeans.js exactly.

scikit-learn's KMeans is used for the published clusters. This separate
implementation exists so the browser can re-cluster for any k and produce the
same labels Python would for the same seed. To make that possible, both sides
use the mulberry32 generator and perform every floating point operation in the
same order (per-dimension accumulation, sequential sums, first index wins on
ties), so results are bit-for-bit identical rather than merely close.
"""

from __future__ import annotations

import numpy as np

_M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & _M32


class Mulberry32:
    def __init__(self, seed: int):
        self.state = seed & _M32

    def __call__(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & _M32
        a = self.state
        t = _imul(a ^ (a >> 15), a | 1)
        t = ((t + _imul(t ^ (t >> 7), t | 61)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0


def _sq_dist_to(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Squared distance from every row of X to c, accumulated dimension by dimension."""
    acc = np.zeros(X.shape[0])
    for j in range(X.shape[1]):
        diff = X[:, j] - c[j]
        acc = acc + diff * diff
    return acc


def _seq_sum(v: np.ndarray) -> float:
    return float(np.cumsum(v)[-1]) if len(v) else 0.0


def _kpp_init(X: np.ndarray, k: int, rng: Mulberry32) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    first = min(int(rng() * n), n - 1)
    centers[0] = X[first]
    closest = _sq_dist_to(X, centers[0])
    for c in range(1, k):
        cum = np.cumsum(closest)
        r = rng() * cum[-1]
        idx = int(np.searchsorted(cum, r, side="right"))
        idx = min(idx, n - 1)
        centers[c] = X[idx]
        closest = np.minimum(closest, _sq_dist_to(X, centers[c]))
    return centers


def _assign(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.stack([_sq_dist_to(X, c) for c in centers], axis=1)
    labels = np.argmin(d, axis=1)  # first minimum wins, same as the JS loop
    return labels, d[np.arange(X.shape[0]), labels]


def kmeans(
    X: np.ndarray, k: int, seed: int = 42, n_init: int = 10, max_iter: int = 300
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's algorithm with k-means++ seeding. Returns (labels, centers, inertia)."""
    X = np.asarray(X, dtype=float)
    rng = Mulberry32(seed)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(n_init):
        centers = _kpp_init(X, k, rng)
        labels = None
        for _ in range(max_iter):
            new_labels, _ = _assign(X, centers)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = X[labels == c]
                if len(members):
                    centers[c] = np.cumsum(members, axis=0)[-1] / len(members)
        labels, mind = _assign(X, centers)
        inertia = _seq_sum(mind)
        if best is None or inertia < best[2]:
            best = (labels.copy(), centers.copy(), inertia)
    assert best is not None
    return best
