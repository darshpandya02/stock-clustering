// K-means++ for the browser. Mirrors src/stockclust/seeded_kmeans.py operation
// for operation (same PRNG, same summation order, first index wins on ties), so
// for a given seed it returns exactly the labels the Python version returns.
// tests/test_js_parity.py checks this.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sqDist(x, c) {
  let acc = 0;
  for (let j = 0; j < x.length; j++) {
    const diff = x[j] - c[j];
    acc = acc + diff * diff;
  }
  return acc;
}

function kppInit(X, k, rng) {
  const n = X.length;
  const centers = [];
  const first = Math.min(Math.floor(rng() * n), n - 1);
  centers.push(X[first].slice());
  const closest = new Float64Array(n);
  for (let i = 0; i < n; i++) closest[i] = sqDist(X[i], centers[0]);
  for (let c = 1; c < k; c++) {
    let total = 0;
    const cum = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      total = total + closest[i];
      cum[i] = total;
    }
    const r = rng() * total;
    let idx = n - 1;
    for (let i = 0; i < n; i++) {
      if (cum[i] > r) {
        idx = i;
        break;
      }
    }
    centers.push(X[idx].slice());
    for (let i = 0; i < n; i++) {
      const d = sqDist(X[i], centers[c]);
      if (d < closest[i]) closest[i] = d;
    }
  }
  return centers;
}

function assign(X, centers) {
  const n = X.length;
  const labels = new Int32Array(n);
  const mind = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    let bestD = sqDist(X[i], centers[0]);
    for (let c = 1; c < centers.length; c++) {
      const d = sqDist(X[i], centers[c]);
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    labels[i] = best;
    mind[i] = bestD;
  }
  return { labels, mind };
}

function sameLabels(a, b) {
  if (!a || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function kmeans(X, k, { seed = 42, nInit = 10, maxIter = 300 } = {}) {
  const rng = mulberry32(seed);
  const dim = X[0].length;
  let best = null;
  for (let run = 0; run < nInit; run++) {
    const centers = kppInit(X, k, rng);
    let labels = null;
    for (let it = 0; it < maxIter; it++) {
      const { labels: newLabels } = assign(X, centers);
      if (sameLabels(labels, newLabels)) break;
      labels = newLabels;
      for (let c = 0; c < k; c++) {
        const sum = new Float64Array(dim);
        let count = 0;
        for (let i = 0; i < X.length; i++) {
          if (labels[i] !== c) continue;
          for (let j = 0; j < dim; j++) sum[j] = sum[j] + X[i][j];
          count++;
        }
        if (count > 0) {
          for (let j = 0; j < dim; j++) centers[c][j] = sum[j] / count;
        }
      }
    }
    const { labels: finalLabels, mind } = assign(X, centers);
    let inertia = 0;
    for (let i = 0; i < mind.length; i++) inertia = inertia + mind[i];
    if (best === null || inertia < best.inertia) {
      best = { labels: Array.from(finalLabels), centers: centers.map((c) => c.slice()), inertia };
    }
  }
  return best;
}

// Mean silhouette coefficient (Euclidean), matching sklearn's definition.
// Points in singleton clusters get a score of 0.
export function silhouette(X, labels) {
  const n = X.length;
  const k = Math.max(...labels) + 1;
  if (k < 2) return null;
  const sizes = new Array(k).fill(0);
  for (const l of labels) sizes[l]++;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const sums = new Array(k).fill(0);
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      sums[labels[j]] += Math.sqrt(sqDist(X[i], X[j]));
    }
    const own = labels[i];
    if (sizes[own] <= 1) continue;
    const a = sums[own] / (sizes[own] - 1);
    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === own || sizes[c] === 0) continue;
      b = Math.min(b, sums[c] / sizes[c]);
    }
    total += (b - a) / Math.max(a, b);
  }
  return total / n;
}
