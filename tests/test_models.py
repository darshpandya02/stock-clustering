import numpy as np
import pytest
from sklearn.datasets import make_blobs

from stockclust import models
from stockclust.seeded_kmeans import Mulberry32, kmeans


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=300, centers=4, n_features=5, cluster_std=0.6, random_state=3)
    return X, y


def test_select_k_finds_blobs(blobs):
    X, y = blobs
    sel = models.select_k(X, k_values=range(2, 8))
    assert sel.best_k == 4
    assert list(sel.table.k) == list(range(2, 8))


def test_kmeans_deterministic(blobs):
    X, _ = blobs
    a = models.fit_kmeans(X, 4, seed=11).labels_
    b = models.fit_kmeans(X, 4, seed=11).labels_
    assert (a == b).all()


def test_kmeans_recovers_blobs(blobs):
    X, y = blobs
    assert models.ari(models.fit_kmeans(X, 4).labels_, y) == pytest.approx(1.0)


def test_dbscan_knee_and_noise(blobs):
    X, y = blobs
    X = np.vstack([X, [[50, 50, 50, 50, 50]]])
    res = models.fit_dbscan(X)
    assert res.min_samples == 10
    assert res.labels[-1] == -1
    assert len(set(res.labels) - {-1}) == 4
    again = models.fit_dbscan(X)
    assert (again.labels == res.labels).all() and again.eps == res.eps


def test_knee_index():
    y = np.array([1, 1.1, 1.2, 1.3, 1.4, 1.5, 5, 10])
    assert models.knee_index(y) == 5


def test_silhouette_none_for_single_cluster(blobs):
    X, _ = blobs
    assert models.silhouette_or_none(X, np.zeros(len(X), int)) is None
    assert models.silhouette_or_none(X, np.r_[np.zeros(len(X) - 1, int), -1]) is None


def test_pca_deterministic(blobs):
    X, _ = blobs
    a = models.fit_pca(X).transform(X)
    b = models.fit_pca(X).transform(X)
    assert np.array_equal(a, b)


def test_mulberry32_known_values():
    # Reference values from the JavaScript implementation in web/kmeans.js.
    rng = Mulberry32(42)
    got = [rng() for _ in range(3)]
    assert got == pytest.approx([0.6011037519201636, 0.44829055899754167, 0.8524657934904099], abs=0)


def test_seeded_kmeans_deterministic_and_close_to_sklearn(blobs):
    X, y = blobs
    la, _, ia = kmeans(X, 4, seed=5)
    lb, _, ib = kmeans(X, 4, seed=5)
    assert (la == lb).all() and ia == ib
    sk = models.fit_kmeans(X, 4)
    assert models.ari(la, sk.labels_) == pytest.approx(1.0)
    assert ia == pytest.approx(sk.inertia_, rel=1e-6)
