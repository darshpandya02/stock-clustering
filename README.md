# Stock Clustering and Movement Prediction

Rebuilt from scratch in 2026. The original 2024 project code was not preserved.

K-means++ and DBSCAN clustering of S&P 500 stocks on price-derived features, PCA for a 2D view, silhouette
scores to choose k, and an honest walk-forward test of a next-week direction classifier. Prices come from
Yahoo Finance (via `yfinance`) and a scheduled GitHub Actions job refreshes everything after each US market close.

**Live app:** https://stock-clustering.vercel.app

The app shows the PCA scatter coloured by cluster, a toggle between K-means++ and DBSCAN, hover details for
every stock, a cluster table with sector mix, ticker search with nearest neighbours, and a "Custom k" mode
that re-runs K-means++ in the browser for any k from 2 to 8.

## Method

**Universe.** The 503 S&P 500 constituents in `data/sp500_constituents.csv` (ticker, name, GICS sector and
sub-industry, from the [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies)
list, taken September 2026). Five years of daily adjusted closes are downloaded in batches of 50 with a pause
between batches, and cached locally as parquet. Stocks with under 253 trading days of history, or with no
valid price on the as-of date, are dropped and listed in the snapshot.

**Features** (14, all computed from prices up to the as-of date only):

| Group | Features |
|---|---|
| Returns (price change rates) | 5, 21, 63, 126 and 252 trading days |
| Momentum change | 21-day return minus the previous 21-day return |
| Volatility | annualized std of daily log returns over 21 and 63 days |
| Moving-average ratios | price / SMA20, price / SMA50, price / SMA200, SMA50 / SMA200 |
| Market sensitivity | 252-day beta vs SPY |
| Risk | 252-day max drawdown |

Each feature is winsorized at the 1st and 99th cross-sectional percentiles, then z-scored.

**K-means++.** scikit-learn `KMeans(init="k-means++", n_init=10, random_state=42)` for k = 2..10. k is the
highest silhouette score among k >= 3. k = 2 always scores highest on this data but only splits off the
most volatile names, so it is reported and not selected.

**DBSCAN.** `min_samples = 2 x features = 28`, and `eps` at the knee of the sorted 28th-neighbour distance
curve (the point farthest below the chord from the first to the last value).

**PCA.** Two components of the standardized features, used for the scatter plot only. Clustering and
nearest neighbours use all 14 dimensions.

**Sector alignment.** Adjusted Rand index (ARI) between cluster labels and GICS sectors, with the ARI of
random labellings as a reference.

**Browser re-clustering.** `web/kmeans.js` and `src/stockclust/seeded_kmeans.py` implement the same
K-means++ with the same mulberry32 PRNG and the same order of floating point operations. For the same seed
they return identical labels and a bit-identical inertia; `tests/test_js_parity.py` runs the JavaScript
under Node and checks this on the live snapshot for several (k, seed) pairs.

**Movement prediction.** Every stock at every 5th trading day becomes one sample, with the same 14 features
z-scored within that week. The target is whether the next 5-day return is positive (and, separately, whether
it beats SPY). Walk-forward, expanding window: after a 52-week warm-up, models are refit every 4 weeks on
earlier weeks only. At each refit K-means (same k) is fit on the training rows, test rows go to the nearest
centroid, and one logistic regression is fit per cluster. Baselines are the training majority class and
"same direction as last week". Stocks in the same week move together, so standard errors are computed from
per-week accuracies rather than treating 73,530 predictions as independent.

## Results

Snapshot with prices through **2026-09-25** (499 stocks included). The live app shows the latest refresh,
so its numbers will drift from these.

| Measure | Value |
|---|---|
| Chosen k (highest silhouette, k >= 3) | 4 |
| K-means++ silhouette at k = 4 | 0.260 |
| K-means++ cluster sizes (largest first) | 285, 116, 79, 19 |
| Silhouette at k = 2 (not chosen) | 0.419 (sizes 408, 91) |
| ARI vs GICS sectors, K-means++ k = 4 | 0.038 (random labels: mean 0.000, 95th pct 0.005) |
| DBSCAN eps (knee), min_samples | 3.61, 28 |
| DBSCAN result | 1 cluster of 466, 33 noise points |
| DBSCAN silhouette | undefined (one cluster) |
| ARI vs GICS sectors, DBSCAN | 0.002 |
| Variance explained by PC1 + PC2 | 67.3% (48.6% + 18.7%) |
| Browser K-means vs scikit-learn at k = 4 | ARI 0.994, inertia 3469.77 vs 3470.03 |

Silhouette by k: 2: 0.419, 3: 0.256, 4: 0.260, 5: 0.201, 6: 0.171, 7: 0.169, 8: 0.168, 9: 0.173, 10: 0.176.

What the clusters are, in this snapshot: a large low-volatility, low-beta group (285; Financials,
Industrials and Health Care lead); a group with negative 12-month returns and deep drawdowns (116); a
group trading well above its moving averages (79, 46% Information Technology); and a small very
high-momentum, high-volatility group (19, 84% Information Technology, median 12-month return 292%).
25 of the 33 DBSCAN noise points are Information Technology stocks, mostly the same high-volatility names.

**Sectors are barely recovered (ARI 0.038).** These features describe recent price behaviour (trend,
volatility, beta), which cuts across sectors. The ARI is well above the random-label reference, but low
in absolute terms.

**DBSCAN finds no group structure.** The standardized feature cloud is one dense core with a sparse tail,
so DBSCAN with a knee-based eps returns a single cluster plus outliers. It works here as an outlier
detector, not as a clustering method.

### Next-week direction (walk-forward, 148 test weeks from 2023-10-12 to 2026-09-18, 73,530 predictions)

| Method | Next week up/down | Beats SPY next week |
|---|---|---|
| Logistic regression per cluster | 53.1% ± 1.3% | 51.4% ± 0.9% |
| Logistic regression, pooled | 53.7% ± 1.5% | 52.0% ± 1.1% |
| Baseline: training majority class | **53.9% ± 1.6%** | **52.0% ± 1.1%** |
| Baseline: same direction as last week | 50.1% ± 0.8% | 49.4% ± 0.5% |

(± one week-clustered standard error.) **The per-cluster classifier does not beat the naive baseline.**
It trails "always predict up" by 0.7 points on average (± 0.3), and was better in 55 weeks and worse in
83. The 53.9% "accuracy" of the baseline is just the share of stock-weeks that went up in a rising
market. On these features, next-week direction is a coin flip.

## Running it

Requires Python 3.11+ with [uv](https://docs.astral.sh/uv/), and Node 18+ for the parity tests.

```bash
uv sync
uv run python -m stockclust.pipeline          # download prices (about 3 minutes), write web/data/snapshot.json
uv run python -m stockclust.pipeline --no-predict   # skip the backtest, keep the previous one
uv run pytest -q                               # features, models, walk-forward, JS/Python parity
python3 -m http.server -d web 8000             # open http://localhost:8000
```

`scripts/verify_site.py` drives a deployed copy in a headless browser (scatter, toggle, search, hover,
re-clustering and a label-for-label comparison with Python):

```bash
uv run --with playwright python scripts/verify_site.py https://stock-clustering.vercel.app
```

**Refresh.** `.github/workflows/refresh.yml` runs on weekdays at 22:30 UTC. It downloads prices, rebuilds
the snapshot and the backtest, runs the tests against the new snapshot, and commits
`web/data/snapshot.json` if anything other than the timestamp changed. Vercel redeploys on the commit. The
site only reads the committed snapshot, so it keeps working when Yahoo Finance is down or rate-limiting;
a failed refresh just leaves the previous snapshot in place, and the page warns when it is over 5 days old.

## Layout

```
src/stockclust/   data.py (download + cache), features.py, models.py, seeded_kmeans.py,
                  prediction.py (walk-forward), pipeline.py (builds the snapshot)
web/              static site: index.html, app.js, kmeans.js, style.css, data/snapshot.json
tests/            pytest suite, plus the Node runner for the JS parity test
scripts/          verify_site.py (browser check)
```

## Limitations

- **Survivorship bias.** The universe is today's S&P 500 list applied to five years of history, so
  stocks that were removed from the index are missing from the backtest.
- **Snapshot clusters are not stable over time.** They describe the trailing year as of one date.
  Cluster numbers can be reassigned between refreshes.
- **Low sector agreement is a property of the features**, not a bug: momentum and volatility features
  do not encode industry. Clustering on return correlations would be the approach for sector-like groups.
- **Yahoo Finance data is unofficial.** `yfinance` scrapes a public endpoint that can change, throttle or
  return gaps, and adjusted prices can be revised.
- **k selection excludes k = 2 by rule.** That choice is documented above and reported, but it is a choice.
- **The prediction test uses one simple model family** (logistic regression on 14 features, one-week
  horizon). It shows these features carry no usable next-week signal in this setup; it does not show that
  no signal exists.

## Not investment advice

This is a data analysis project. Nothing here is a recommendation to buy or sell any security, and the
backtest shows the direction classifier does not beat a trivial baseline.
