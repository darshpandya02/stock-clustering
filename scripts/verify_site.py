"""Browser check of a deployed (or local) copy of the site.

    uv run --with playwright --python 3.12 python scripts/verify_site.py https://example.vercel.app

Checks that the scatter renders, the model toggle works, ticker search works,
hovering shows a tooltip, and that re-clustering in the browser gives the same
partition as the Python implementation (read from the snapshot the page loads).
"""

import json
import re
import sys
import urllib.request

from playwright.sync_api import sync_playwright

url = sys.argv[1].rstrip("/")
k, seed = 6, 42
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


with urllib.request.urlopen(url + "/") as r:
    check("GET / unauthenticated", r.status == 200, f"status={r.status}")
with urllib.request.urlopen(url + "/data/snapshot.json") as r:
    snap = json.loads(r.read())
    check("GET /data/snapshot.json", r.status == 200, f"as_of={snap['as_of']} stocks={len(snap['stocks'])}")

expected = None
try:
    import numpy as np

    sys.path.insert(0, "src")
    from stockclust.seeded_kmeans import kmeans

    expected = kmeans(np.array([s["z"] for s in snap["stocks"]]), k, seed=seed, n_init=10)[0].tolist()
except ImportError:
    pass

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url + "/")
    page.wait_for_selector("body[data-ready=true]", timeout=30000)

    n_pts = page.locator("#scatter path.pt").count()
    check("scatter renders every stock", n_pts == len(snap["stocks"]), f"points={n_pts}")
    check("default is K-means", f"k = {snap['kmeans']['k']}" in page.inner_text("#chart-title"), page.inner_text("#chart-title"))
    check("freshness shows as-of date", snap["as_of"] in page.inner_text("#freshness"))

    box = page.locator("#scatter").bounding_box()
    pts = page.eval_on_selector_all("#scatter path.pt", "els => els.map(e => e.getAttribute('transform'))")
    x, y = map(float, re.findall(r"[-\d.]+", pts[len(pts) // 2]))
    vb = page.eval_on_selector("#scatter", "e => [e.viewBox.baseVal.width, e.viewBox.baseVal.height]")
    page.mouse.move(box["x"] + x * box["width"] / vb[0], box["y"] + y * box["height"] / vb[1])
    tip = page.locator("#tooltip")
    check("hover shows tooltip", tip.is_visible() and len(tip.inner_text()) > 10, tip.inner_text().split("\n")[0] if tip.is_visible() else "")

    page.click("#mode-dbscan")
    noise = page.locator("#scatter path.pt.noise").count()
    check("toggle to DBSCAN", "DBSCAN" in page.inner_text("#chart-title") and noise == snap["dbscan"]["n_noise"], f"noise points={noise}")
    page.click("#mode-kmeans")
    check("toggle back to K-means", page.locator("#scatter path.pt.noise").count() == 0)

    page.fill("#search-input", "NVDA")
    page.press("#search-input", "Enter")
    page.wait_for_selector("#detail-ticker")
    nbrs = page.locator("#neighbours tr").count() - 1
    check("search shows ticker, cluster and neighbours", page.inner_text("#detail-ticker") == "NVDA" and nbrs == 5,
          f"{page.inner_text('#detail-cluster')}; neighbours={nbrs}")

    page.fill("#k-input", str(k))
    page.fill("#seed-input", str(seed))
    page.click("#recluster-btn")
    page.wait_for_function("document.querySelector('#chart-title').textContent.includes('browser')")
    fills = page.eval_on_selector_all(
        "#scatter path.pt", "els => els.map(e => [e.dataset.t, e.style.fill])"
    )
    legend = page.locator("#legend button").count()
    check("re-cluster in browser", legend == k, f"legend entries={legend}; {page.inner_text('#recluster-status')}")
    if expected is not None:
        label_of = {t: int(re.search(r"--c8-(\d+)", f).group(1)) for t, f in fills}
        got = [label_of[s["t"]] for s in snap["stocks"]]
        check("browser labels equal Python labels (same seed)", got == expected,
              f"{sum(a == b for a, b in zip(got, expected))}/{len(got)} match")
    page.screenshot(path="/tmp/stock-clustering-verify.png" if len(sys.argv) < 3 else sys.argv[2], full_page=True)
    check("no page errors", not errors, "; ".join(errors))
    browser.close()

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
