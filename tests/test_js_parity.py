"""The browser k-means (web/kmeans.js) must match the Python one exactly for a fixed seed."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import silhouette_score

from stockclust.seeded_kmeans import kmeans

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "kmeans_runner.mjs"
SNAPSHOT = ROOT / "web" / "data" / "snapshot.json"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_js(X, k, seed, n_init, with_silhouette=False):
    payload = json.dumps({"X": X, "k": k, "seed": seed, "nInit": n_init, "withSilhouette": with_silhouette})
    out = subprocess.run(["node", str(RUNNER)], input=payload, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def snapshot_matrix():
    snap = json.loads(SNAPSHOT.read_text())
    return [s["z"] for s in snap["stocks"]]


@pytest.mark.parametrize("k,seed", [(2, 42), (4, 42), (6, 7), (9, 123)])
def test_snapshot_labels_identical(k, seed):
    X = snapshot_matrix()
    py_labels, _, py_inertia = kmeans(np.array(X), k, seed=seed, n_init=10)
    js = run_js(X, k, seed, 10)
    assert js["labels"] == py_labels.tolist()
    assert js["inertia"] == py_inertia  # bit-identical, not approximately equal


def test_synthetic_blobs_identical():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, size=(5, 3))
    X = np.vstack([c + rng.normal(0, 1, size=(40, 3)) for c in centers]).round(6)
    py_labels, _, py_inertia = kmeans(X, 5, seed=1, n_init=5)
    js = run_js(X.tolist(), 5, 1, 5)
    assert js["labels"] == py_labels.tolist()
    assert js["inertia"] == py_inertia


def test_js_silhouette_matches_sklearn():
    X = snapshot_matrix()
    js = run_js(X, 4, 42, 10, with_silhouette=True)
    expected = silhouette_score(np.array(X), js["labels"])
    assert js["silhouette"] == pytest.approx(expected, abs=1e-9)
