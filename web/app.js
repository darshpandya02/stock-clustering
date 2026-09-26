import { kmeans, silhouette } from "./kmeans.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

const state = {
  snap: null,
  mode: "kmeans",
  custom: null,
  selected: null,
  focus: null, // cluster id highlighted from the legend or table
};

// ---------- formatting ----------
const pct = (v, d = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 2) => (v == null ? "n/a" : v.toFixed(d));
const fmtFeature = (name, v) => (name === "beta_252d" ? num(v) : pct(v));
const SHORT = {
  ret_5d: "1w return", ret_21d: "1m return", ret_63d: "3m return", ret_126d: "6m return", ret_252d: "12m return",
  roc_accel_21d: "1m momentum change", vol_21d: "1m volatility", vol_63d: "3m volatility",
  px_vs_sma20: "price vs SMA20", px_vs_sma50: "price vs SMA50", px_vs_sma200: "price vs SMA200",
  sma50_vs_sma200: "SMA50 vs SMA200", beta_252d: "beta", max_drawdown_252d: "drawdown",
};

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else e.setAttribute(k, v);
  }
  if (text != null) e.textContent = text;
  return e;
}
function sv(tag, attrs = {}) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

// ---------- cluster encoding: colour + shape ----------
// Up to 4 clusters use a 4-hue subset that stays distinguishable for every pair
// (checked for colour-vision deficiencies in light and dark mode). Beyond 4 the
// full 8-hue set is used and the shape carries identity as well.
function clusterColor(c, n) {
  if (c < 0) return "var(--noise)";
  return n <= 4 ? `var(--c4-${c})` : `var(--c8-${c % 8})`;
}
const SHAPES = ["circle", "square", "triangle", "diamond", "triangleDown", "plus", "hexagon", "star"];
function shapePath(shape, r) {
  const poly = (pts) => "M" + pts.map((p) => p.map((v) => v.toFixed(2)).join(",")).join("L") + "Z";
  const ngon = (n, rot, rr = r) => Array.from({ length: n }, (_, i) => {
    const a = rot + (i * 2 * Math.PI) / n;
    return [rr * Math.cos(a), rr * Math.sin(a)];
  });
  switch (shape) {
    case "square": { const s = r * 0.88; return poly([[-s, -s], [s, -s], [s, s], [-s, s]]); }
    case "triangle": return poly(ngon(3, -Math.PI / 2, r * 1.2));
    case "triangleDown": return poly(ngon(3, Math.PI / 2, r * 1.2));
    case "diamond": return poly(ngon(4, 0, r * 1.2));
    case "hexagon": return poly(ngon(6, 0, r * 1.05));
    case "plus": {
      const a = r * 1.1, b = r * 0.38;
      return poly([[-b, -a], [b, -a], [b, -b], [a, -b], [a, b], [b, b], [b, a], [-b, a], [-b, b], [-a, b], [-a, -b], [-b, -b]]);
    }
    case "star": {
      const pts = [];
      for (let i = 0; i < 10; i++) {
        const rr = i % 2 ? r * 0.5 : r * 1.25, a = -Math.PI / 2 + (i * Math.PI) / 5;
        pts.push([rr * Math.cos(a), rr * Math.sin(a)]);
      }
      return poly(pts);
    }
    default: return `M${r},0A${r},${r} 0 1,1 ${-r},0A${r},${r} 0 1,1 ${r},0Z`;
  }
}
function clusterShape(c) { return c < 0 ? "circle" : SHAPES[c % SHAPES.length]; }
function swatch(c, n) {
  const s = sv("svg", { viewBox: "-7 -7 14 14", "aria-hidden": "true" });
  const p = sv("path", { d: shapePath(clusterShape(c), 5) });
  if (c < 0) { p.setAttribute("fill", "none"); p.style.stroke = "var(--noise)"; p.setAttribute("stroke-width", "1.5"); }
  else p.style.fill = clusterColor(c, n);
  s.appendChild(p);
  return s;
}
const clusterName = (c) => (c < 0 ? "Noise" : `Cluster ${c + 1}`);

// ---------- metrics ----------
function ari(a, b) {
  const n = a.length;
  const ka = new Map(), kb = new Map(), cont = new Map();
  for (let i = 0; i < n; i++) {
    ka.set(a[i], (ka.get(a[i]) || 0) + 1);
    kb.set(b[i], (kb.get(b[i]) || 0) + 1);
    const key = `${a[i]}\u0000${b[i]}`;
    cont.set(key, (cont.get(key) || 0) + 1);
  }
  const c2 = (x) => (x * (x - 1)) / 2;
  let sumC = 0, sumA = 0, sumB = 0;
  for (const v of cont.values()) sumC += c2(v);
  for (const v of ka.values()) sumA += c2(v);
  for (const v of kb.values()) sumB += c2(v);
  const expected = (sumA * sumB) / c2(n);
  const max = (sumA + sumB) / 2;
  return max === expected ? 1 : (sumC - expected) / (max - expected);
}
function dist(a, b) {
  let s = 0;
  for (let j = 0; j < a.length; j++) { const d = a[j] - b[j]; s += d * d; }
  return Math.sqrt(s);
}
const median = (arr) => {
  const s = arr.filter((v) => v != null).sort((x, y) => x - y);
  if (!s.length) return null;
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// ---------- current labelling ----------
function currentLabels() {
  const { snap, mode, custom } = state;
  if (mode === "dbscan") return snap.stocks.map((s) => s.db);
  if (mode === "custom" && custom) return custom.labels;
  return snap.stocks.map((s) => s.km);
}
function nClusters(labels) { return Math.max(...labels) + 1; }
function modeTitle() {
  const { snap, mode, custom } = state;
  if (mode === "dbscan") return `DBSCAN (eps ${snap.dbscan.eps}, min_samples ${snap.dbscan.min_samples})`;
  if (mode === "custom" && custom) return `K-means++ in your browser (k = ${custom.k}, seed ${custom.seed})`;
  return `K-means++ (k = ${snap.kmeans.k}, chosen by silhouette)`;
}

// ---------- stats ----------
function renderStats() {
  const { snap, mode, custom } = state;
  const box = $("stats");
  box.replaceChildren();
  const add = (v, l) => { const d = el("div", { class: "stat" }); d.append(el("div", { class: "v" }, v), el("div", { class: "l" }, l)); box.append(d); };
  add(String(snap.universe.n_included), `stocks (of ${snap.universe.n_constituents} constituents)`);
  if (mode === "dbscan") {
    const d = snap.dbscan;
    add(String(d.n_clusters), d.n_clusters === 1 ? "dense cluster" : "clusters");
    add(String(d.n_noise), "noise points (outliers)");
    add(d.silhouette == null ? "n/a" : num(d.silhouette, 3), "silhouette (needs 2+ clusters)");
    add(num(d.sector_ari, 3), "ARI vs GICS sectors");
  } else if (mode === "custom" && custom) {
    add(String(custom.k), "clusters");
    add(num(custom.silhouette, 3), "silhouette");
    add(num(custom.ari, 3), "ARI vs GICS sectors");
    add(num(custom.inertia, 1), "inertia");
  } else {
    const k = snap.kmeans;
    add(String(k.k), "clusters");
    add(num(k.silhouette, 3), "silhouette");
    add(num(k.sector_ari, 3), `ARI vs GICS sectors (random: ${num(k.random_ari_mean, 3)})`);
    add(pct(snap.pca.explained_variance_ratio[0] + snap.pca.explained_variance_ratio[1], 0), "variance shown by PC1 + PC2");
  }
}

// ---------- scatter ----------
const scatter = { pts: [], sx: null, sy: null };

function niceTicks(lo, hi, count = 6) {
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function renderScatter() {
  const { snap } = state;
  const svg = $("scatter");
  const labels = currentLabels();
  const n = nClusters(labels);
  const W = svg.clientWidth || 800, H = svg.clientHeight || 500;
  const m = { l: 44, r: 16, t: 12, b: 40 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();

  const xs = snap.stocks.map((s) => s.pca[0]), ys = snap.stocks.map((s) => s.pca[1]);
  const pad = (a, b) => [(a - (b - a) * 0.04), (b + (b - a) * 0.04)];
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs));
  const [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  scatter.sx = sx; scatter.sy = sy;

  const g = sv("g", { class: "axis" });
  for (const t of niceTicks(x0, x1)) {
    g.append(sv("line", { class: "gridline", x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b }));
    const tx = sv("text", { x: sx(t), y: H - m.b + 16, "text-anchor": "middle" }); tx.textContent = t; g.append(tx);
  }
  for (const t of niceTicks(y0, y1)) {
    g.append(sv("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) }));
    const ty = sv("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end" }); ty.textContent = t; g.append(ty);
  }
  const evr = snap.pca.explained_variance_ratio;
  const xt = sv("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" });
  xt.textContent = `PC1 (${pct(evr[0], 0)} of variance)`;
  const yt = sv("text", { class: "axis-title", x: -(m.t + H - m.b) / 2, y: 12, transform: "rotate(-90)", "text-anchor": "middle" });
  yt.textContent = `PC2 (${pct(evr[1], 0)} of variance)`;
  g.append(xt, yt);
  svg.append(g);

  const layer = sv("g");
  scatter.pts = [];
  // Draw larger clusters first so small ones stay visible on top; noise first of all.
  const order = snap.stocks.map((_, i) => i);
  const sizes = {};
  labels.forEach((l) => (sizes[l] = (sizes[l] || 0) + 1));
  order.sort((a, b) => (labels[a] < 0 ? -1 : 0) - (labels[b] < 0 ? -1 : 0) || sizes[labels[b]] - sizes[labels[a]]);
  for (const i of order) {
    const s = snap.stocks[i], c = labels[i];
    const x = sx(s.pca[0]), y = sy(s.pca[1]);
    const p = sv("path", { d: shapePath(clusterShape(c), 4.2), transform: `translate(${x.toFixed(1)},${y.toFixed(1)})`, class: c < 0 ? "pt noise" : "pt", "data-t": s.t });
    if (c >= 0) p.style.fill = clusterColor(c, n);
    if (state.focus != null && state.focus !== c) p.classList.add("dim");
    layer.append(p);
    scatter.pts.push({ i, x, y });
  }
  svg.append(layer);

  // Direct labels at each cluster's median position.
  const lab = sv("g");
  for (let c = 0; c < n; c++) {
    const idx = labels.map((l, i) => (l === c ? i : -1)).filter((i) => i >= 0);
    if (!idx.length) continue;
    const mx = median(idx.map((i) => snap.stocks[i].pca[0])), my = median(idx.map((i) => snap.stocks[i].pca[1]));
    const t = sv("text", { class: "clabel", x: sx(mx), y: sy(my) - 8, "text-anchor": "middle" });
    t.textContent = `C${c + 1}`;
    lab.append(t);
  }
  svg.append(lab);

  const overlay = sv("g", { id: "overlay" });
  svg.append(overlay);
  drawSelection();

  $("chart-title").textContent = modeTitle();
  renderLegend(labels, n);
  $("pca-note").textContent = "Each point is a stock. Position comes from PCA of the 14 standardized features, so nearby points had similar price behaviour. Hover for details, click to select.";
}

function drawSelection() {
  const overlay = $("overlay");
  if (!overlay) return;
  overlay.replaceChildren();
  if (!state.selected) return;
  const { snap } = state;
  const i = snap.stocks.findIndex((s) => s.t === state.selected);
  if (i < 0) return;
  for (const nb of neighbours(i, 5)) {
    const s = snap.stocks[nb.j];
    overlay.append(sv("circle", { class: "nbr-ring", cx: scatter.sx(s.pca[0]), cy: scatter.sy(s.pca[1]), r: 8 }));
  }
  const s = snap.stocks[i];
  const cx = scatter.sx(s.pca[0]), cy = scatter.sy(s.pca[1]);
  overlay.append(sv("circle", { class: "ring", cx, cy, r: 9 }));
  const t = sv("text", { class: "sel-label", x: cx + 12, y: cy + 4 });
  t.textContent = s.t;
  overlay.append(t);
}

function renderLegend(labels, n) {
  const box = $("legend");
  box.replaceChildren();
  const ids = [...new Set(labels)].sort((a, b) => a - b);
  for (const c of ids) {
    const b = el("button", { type: "button", "aria-pressed": String(state.focus === c) });
    b.append(swatch(c, n), document.createTextNode(clusterName(c)));
    b.addEventListener("click", () => { state.focus = state.focus === c ? null : c; renderScatter(); });
    box.append(b);
  }
}

function nearestPoint(px, py, maxDist = 24) {
  let best = null, bd = maxDist * maxDist;
  for (const p of scatter.pts) {
    const d = (p.x - px) ** 2 + (p.y - py) ** 2;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

function showTooltip(tip, wrap, px, py, build) {
  tip.replaceChildren();
  build(tip);
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight, W = wrap.clientWidth;
  let left = px + 14, top = py + 14;
  if (left + w > W) left = px - w - 14;
  if (top + h > wrap.clientHeight) top = Math.max(0, py - h - 14);
  tip.style.left = `${Math.max(0, left)}px`;
  tip.style.top = `${top}px`;
}

function stockTooltip(tip, i) {
  const { snap } = state;
  const s = snap.stocks[i];
  const labels = currentLabels();
  const f = snap.features;
  tip.append(el("div", { class: "tt-head" }, `${s.t}  ${s.name}`), el("div", { class: "tt-sub" }, `${s.sector} · ${clusterName(labels[i])}`));
  const tbl = el("table");
  for (const name of ["ret_252d", "ret_63d", "ret_21d", "vol_63d", "beta_252d", "max_drawdown_252d", "px_vs_sma200"]) {
    const tr = el("tr");
    tr.append(el("td", { class: "v" }, fmtFeature(name, s.raw[f.indexOf(name)])), el("td", {}, SHORT[name]));
    tbl.append(tr);
  }
  tip.append(tbl);
}

function wireScatter() {
  const svg = $("scatter"), tip = $("tooltip"), wrap = $("chart-wrap");
  let hoverRing = null;
  svg.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const vb = svg.viewBox.baseVal;
    const px = ((ev.clientX - r.left) / r.width) * vb.width, py = ((ev.clientY - r.top) / r.height) * vb.height;
    const p = nearestPoint(px, py);
    if (hoverRing) { hoverRing.remove(); hoverRing = null; }
    if (!p) { tip.hidden = true; svg.style.cursor = "default"; return; }
    svg.style.cursor = "pointer";
    hoverRing = sv("circle", { class: "hover-ring", cx: p.x, cy: p.y, r: 7 });
    svg.append(hoverRing);
    showTooltip(tip, wrap, ev.clientX - r.left, ev.clientY - r.top, (t) => stockTooltip(t, p.i));
  });
  svg.addEventListener("pointerleave", () => { tip.hidden = true; if (hoverRing) { hoverRing.remove(); hoverRing = null; } });
  svg.addEventListener("click", (ev) => {
    const r = svg.getBoundingClientRect();
    const vb = svg.viewBox.baseVal;
    const p = nearestPoint(((ev.clientX - r.left) / r.width) * vb.width, ((ev.clientY - r.top) / r.height) * vb.height);
    if (p) selectTicker(state.snap.stocks[p.i].t);
  });
}

// ---------- neighbours & detail ----------
function neighbours(i, k) {
  const { snap } = state;
  const zi = snap.stocks[i].z;
  return snap.stocks
    .map((s, j) => ({ j, d: j === i ? Infinity : dist(zi, s.z) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, k);
}

function selectTicker(t) {
  state.selected = t;
  history.replaceState(null, "", `#${encodeURIComponent(t)}`);
  renderDetail();
  drawSelection();
}

function renderDetail() {
  const { snap } = state;
  const box = $("detail");
  box.replaceChildren(el("h2", {}, "Ticker detail"));
  const i = snap.stocks.findIndex((s) => s.t === state.selected);
  if (i < 0) {
    box.append(el("p", { class: "muted" }, "Search for a ticker, or click a point on the chart, to see its cluster, features and nearest neighbours."));
    return;
  }
  const s = snap.stocks[i];
  const labels = currentLabels();
  const n = nClusters(labels);
  box.append(el("h3", { id: "detail-ticker" }, s.t), el("div", { class: "sub" }, `${s.name} · ${s.sector} · ${s.industry}`));
  const pill = el("span", { class: "pill", id: "detail-cluster" });
  pill.append(swatch(labels[i], n), document.createTextNode(`${clusterName(labels[i])} in ${state.mode === "dbscan" ? "DBSCAN" : state.mode === "custom" && state.custom ? `custom k=${state.custom.k}` : "K-means++"}`));
  box.append(pill);
  const other = el("p", { class: "muted small" },
    `K-means++: ${clusterName(s.km)}. DBSCAN: ${s.db < 0 ? "noise (outlier)" : clusterName(s.db)}.`);
  box.append(other);

  box.append(el("h2", {}, "Nearest neighbours"));
  box.append(el("p", { class: "muted small" }, "Closest stocks by Euclidean distance on the 14 standardized features (not on the 2D plot)."));
  const nt = el("table", { id: "neighbours" });
  const hr = el("tr");
  ["Ticker", "Sector", "Cluster", "Distance"].forEach((h, k) => hr.append(el("th", k === 3 ? { class: "num" } : {}, h)));
  nt.append(hr);
  for (const nb of neighbours(i, 5)) {
    const o = snap.stocks[nb.j];
    const tr = el("tr", { class: "clickable", title: o.name });
    tr.append(el("td", {}, o.t), el("td", {}, o.sector));
    const tdc = el("td"); const sw = el("span", { class: "swatch" }); sw.append(swatch(labels[nb.j], n), document.createTextNode(labels[nb.j] < 0 ? "Noise" : `C${labels[nb.j] + 1}`)); tdc.append(sw);
    tr.append(tdc, el("td", { class: "num" }, num(nb.d)));
    tr.addEventListener("click", () => selectTicker(o.t));
    nt.append(tr);
  }
  box.append(nt);

  box.append(el("h2", {}, "Features"));
  const ft = el("table", { class: "compact" });
  const fh = el("tr");
  ["Feature", "Value", "z-score"].forEach((h, k) => fh.append(el("th", k ? { class: "num" } : {}, h)));
  ft.append(fh);
  snap.features.forEach((name, j) => {
    const tr = el("tr");
    tr.append(el("td", {}, snap.feature_labels[name]), el("td", { class: "num" }, fmtFeature(name, s.raw[j])), el("td", { class: "num" }, num(s.z[j])));
    ft.append(tr);
  });
  box.append(ft);
}

// ---------- cluster table ----------
function describeCluster(idx) {
  const { snap } = state;
  const d = snap.features.length;
  const mean = new Array(d).fill(0);
  for (const i of idx) for (let j = 0; j < d; j++) mean[j] += snap.stocks[i].z[j] / idx.length;
  return mean
    .map((v, j) => ({ v, j }))
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .slice(0, 2)
    .map(({ v, j }) => {
      const name = snap.features[j];
      // Drawdown is stored as a negative number, so a high value means a shallow drawdown.
      if (name === "max_drawdown_252d") return v > 0 ? "shallow drawdown" : "deep drawdown";
      return `${v > 0 ? "high" : "low"} ${SHORT[name]}`;
    })
    .join(", ");
}

function renderClusterTable() {
  const { snap } = state;
  const labels = currentLabels();
  const n = nClusters(labels);
  const f = snap.features;
  const tbl = $("cluster-table");
  tbl.replaceChildren();
  const hr = el("tr");
  const heads = ["Cluster", "Stocks", "Stands out for", "Sector mix (top 3)", "12m return", "3m volatility", "Beta", "Max drawdown", "Examples"];
  heads.forEach((h, k) => hr.append(el("th", k === 1 || (k >= 4 && k <= 7) ? { class: "num" } : {}, h)));
  tbl.append(hr);
  const ids = [...new Set(labels)].sort((a, b) => (a < 0) - (b < 0) || a - b);
  for (const c of ids) {
    const idx = labels.map((l, i) => (l === c ? i : -1)).filter((i) => i >= 0);
    const tr = el("tr", { class: "clickable" });
    const tdc = el("td"); const sw = el("span", { class: "swatch" }); sw.append(swatch(c, n), document.createTextNode(clusterName(c))); tdc.append(sw);
    tr.append(tdc, el("td", { class: "num" }, String(idx.length)));
    tr.append(el("td", {}, c < 0 ? "outliers: too sparse to join a cluster" : describeCluster(idx)));
    const counts = {};
    idx.forEach((i) => (counts[snap.stocks[i].sector] = (counts[snap.stocks[i].sector] || 0) + 1));
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3);
    tr.append(el("td", {}, top.map(([s, k]) => `${s} ${Math.round((100 * k) / idx.length)}%`).join(", ")));
    for (const name of ["ret_252d", "vol_63d", "beta_252d", "max_drawdown_252d"]) {
      tr.append(el("td", { class: "num" }, fmtFeature(name, median(idx.map((i) => snap.stocks[i].raw[f.indexOf(name)])))));
    }
    const ex = idx.map((i) => snap.stocks[i]).sort((a, b) => a.t.localeCompare(b.t));
    tr.append(el("td", { class: "small" }, ex.slice(0, 8).map((s) => s.t).join(" ") + (ex.length > 8 ? " ..." : "")));
    tr.addEventListener("click", () => { state.focus = state.focus === c ? null : c; renderScatter(); });
    tbl.append(tr);
  }
  $("cluster-note").textContent = "Medians of raw feature values. Click a row to highlight that cluster on the chart.";
}

// ---------- small charts ----------
function lineChart(svg, points, { xLabel, yLabel, fmtY, marker, hline, tooltip }) {
  const W = svg.clientWidth || 500, H = svg.clientHeight || 220;
  const m = { l: 48, r: 12, t: 10, b: 34 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(0, ...ys), y1 = Math.max(...ys) * 1.08;
  const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const g = sv("g", { class: "axis" });
  for (const t of niceTicks(y0, y1, 4)) {
    g.append(sv("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) }));
    const tx = sv("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end" }); tx.textContent = fmtY(t); g.append(tx);
  }
  const xt = points.length <= 12 ? xs : niceTicks(x0, x1, 5);
  for (const t of xt) { const tx = sv("text", { x: sx(t), y: H - m.b + 15, "text-anchor": "middle" }); tx.textContent = t; g.append(tx); }
  g.append(sv("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: sy(y0), y2: sy(y0) }));
  const a = sv("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle" }); a.textContent = xLabel; g.append(a);
  const b = sv("text", { class: "axis-title", x: -(m.t + H - m.b) / 2, y: 11, transform: "rotate(-90)", "text-anchor": "middle" }); b.textContent = yLabel; g.append(b);
  svg.append(g);
  if (hline) {
    svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: sy(hline.y), y2: sy(hline.y), stroke: "var(--muted)", "stroke-dasharray": "4 3" }));
    const t = sv("text", { class: "axis-title", x: m.l + 6, y: sy(hline.y) - 5 }); t.textContent = hline.label; svg.append(t);
  }
  const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
  svg.append(sv("path", { d, fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
  if (points.length <= 12) for (const p of points) svg.append(sv("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, fill: "var(--accent)", stroke: "var(--surface-1)", "stroke-width": 2 }));
  if (marker) {
    svg.append(sv("circle", { cx: sx(marker.x), cy: sy(marker.y), r: 7, fill: "none", stroke: "var(--text-primary)", "stroke-width": 2 }));
    const right = sx(marker.x) > W * 0.6;
    const t = sv("text", { class: "sel-label", x: sx(marker.x) + (right ? -10 : 10), y: sy(marker.y) - 8, "text-anchor": right ? "end" : "start" });
    t.textContent = marker.label;
    svg.append(t);
  }
  // Crosshair tooltip that snaps to the nearest x.
  const wrap = svg.parentElement, tip = wrap.querySelector(".tooltip");
  const cross = sv("line", { stroke: "var(--axis)", y1: m.t, y2: H - m.b, visibility: "hidden" });
  svg.append(cross);
  svg.onpointermove = (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    let best = points[0];
    for (const p of points) if (Math.abs(sx(p.x) - px) < Math.abs(sx(best.x) - px)) best = p;
    cross.setAttribute("x1", sx(best.x)); cross.setAttribute("x2", sx(best.x)); cross.setAttribute("visibility", "visible");
    showTooltip(tip, wrap, ev.clientX - r.left, ev.clientY - r.top, (t) => tooltip(t, best));
  };
  svg.onpointerleave = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); };
}

function renderSmallCharts() {
  const { snap } = state;
  const km = snap.kmeans;
  const pts = km.selection.map((r) => ({ x: r.k, y: r.silhouette, inertia: r.inertia }));
  const best = pts.find((p) => p.x === km.k);
  lineChart($("sil-chart"), pts, {
    xLabel: "k (number of clusters)", yLabel: "silhouette", fmtY: (v) => v.toFixed(2),
    marker: { x: best.x, y: best.y, label: `chosen k = ${km.k}` },
    tooltip: (t, p) => { t.append(el("div", { class: "tt-head" }, `k = ${p.x}`)); const tb = el("table"); const r1 = el("tr"); r1.append(el("td", { class: "v" }, p.y.toFixed(4)), el("td", {}, "silhouette")); const r2 = el("tr"); r2.append(el("td", { class: "v" }, p.inertia.toFixed(0)), el("td", {}, "inertia")); tb.append(r1, r2); t.append(tb); },
  });
  const st = $("sil-table");
  st.replaceChildren();
  const h = el("tr"); h.append(el("th", {}, "k"), el("th", { class: "num" }, "silhouette"), el("th", { class: "num" }, "inertia")); st.append(h);
  for (const r of km.selection) { const tr = el("tr"); tr.append(el("td", {}, String(r.k) + (r.k === km.k ? " (chosen)" : "")), el("td", { class: "num" }, r.silhouette.toFixed(4)), el("td", { class: "num" }, r.inertia.toFixed(0))); st.append(tr); }

  const db = snap.dbscan;
  const kp = db.kdist.map((v, i) => ({ x: i + 1, y: v }));
  lineChart($("kdist-chart"), kp, {
    xLabel: "stocks, sorted by k-distance", yLabel: `distance to ${db.min_samples}th neighbour`, fmtY: (v) => v.toFixed(1),
    marker: { x: db.knee_index + 1, y: db.eps, label: `knee: eps = ${db.eps}` },
    tooltip: (t, p) => { t.append(el("div", { class: "tt-head" }, `rank ${p.x}`)); const tb = el("table"); const r1 = el("tr"); r1.append(el("td", { class: "v" }, p.y.toFixed(3)), el("td", {}, "k-distance")); tb.append(r1); t.append(tb); },
  });
  $("kdist-note").textContent = `min_samples = ${db.min_samples} (2 x features). eps is the knee of the sorted distance to each stock's ${db.min_samples}th nearest neighbour. Result: ${db.n_clusters} dense cluster${db.n_clusters === 1 ? "" : "s"} of ${db.sizes.join(", ")} stocks and ${db.n_noise} noise points. The feature cloud has no low-density gaps, so DBSCAN acts as an outlier detector here rather than finding groups.`;
}

// ---------- prediction ----------
function renderPrediction() {
  const { snap } = state;
  const box = $("prediction-body");
  box.replaceChildren();
  const p = snap.prediction;
  if (!p) { box.append(el("p", { class: "muted" }, "No backtest in this snapshot.")); return; }
  const raw = p.raw, ex = p.excess;
  const beat = raw.accuracy.per_cluster_logit.accuracy > raw.accuracy.majority.accuracy;
  box.append(el("p", { class: "callout", id: "prediction-verdict" },
    beat
      ? `The per-cluster classifier was right ${pct(raw.accuracy.per_cluster_logit.accuracy)} of the time versus ${pct(raw.accuracy.majority.accuracy)} for always predicting the majority class. The gap is small; treat it as noise unless it holds up over more data.`
      : `The per-cluster classifier did not beat the naive baseline: ${pct(raw.accuracy.per_cluster_logit.accuracy)} accuracy versus ${pct(raw.accuracy.majority.accuracy)} for always predicting "up". Next-week direction from these features is roughly a coin flip.`));
  box.append(el("p", { class: "muted small" },
    `Every stock, every week from ${raw.test_start} to ${raw.test_end} (${raw.n_weeks} weeks, ${raw.n_predictions.toLocaleString()} predictions). Models are refit every ${p.refit_every_weeks} weeks on earlier weeks only, after a ${p.initial_weeks}-week warm-up. K-means (k = ${p.k}) is refit on the training rows each time and one logistic regression is fit per cluster. Standard errors are computed across weeks, because stocks in the same week move together.`));
  const names = { per_cluster_logit: "Logistic regression per cluster", pooled_logit: "Logistic regression, all stocks pooled", majority: "Baseline: majority class of training weeks", persistence: "Baseline: same direction as last week" };
  const tbl = el("table", { id: "prediction-table" });
  const hr = el("tr");
  ["Method", "Next-week up or down", "Beats SPY next week"].forEach((h, k) => hr.append(el("th", k ? { class: "num" } : {}, h)));
  tbl.append(hr);
  for (const m of Object.keys(names)) {
    const tr = el("tr");
    const cell = (s) => `${pct(s.accuracy[m].accuracy)} ± ${pct(s.accuracy[m].weekly_se)}`;
    tr.append(el("td", {}, names[m]), el("td", { class: "num" }, cell(raw)), el("td", { class: "num" }, cell(ex)));
    tbl.append(tr);
  }
  box.append(el("div", { class: "table-scroll" }, null));
  box.lastChild.append(tbl);
  box.append(el("p", { class: "muted small" },
    `Share of stock-weeks that went up: ${pct(raw.up_rate)}. Share that beat SPY: ${pct(ex.up_rate)}. Values are accuracy ± one week-clustered standard error.`));
}

// ---------- re-clustering ----------
function recluster() {
  const k = Math.max(2, Math.min(8, parseInt($("k-input").value, 10) || 4));
  const seed = Math.max(0, parseInt($("seed-input").value, 10) || 0);
  $("k-input").value = k;
  $("seed-input").value = seed;
  const X = state.snap.stocks.map((s) => s.z);
  const t0 = performance.now();
  const res = kmeans(X, k, { seed, nInit: 10 });
  const sil = silhouette(X, res.labels);
  const ms = performance.now() - t0;
  state.custom = { k, seed, labels: res.labels, inertia: res.inertia, silhouette: sil, ari: ari(state.snap.stocks.map((s) => s.sector), res.labels) };
  $("recluster-status").textContent = `k = ${k}, seed ${seed}: done in ${ms.toFixed(0)} ms`;
  setMode("custom");
}

function setMode(mode) {
  if (mode === "custom" && !state.custom) { recluster(); return; }
  state.mode = mode;
  state.focus = null;
  for (const b of document.querySelectorAll(".seg button")) b.setAttribute("aria-checked", String(b.dataset.mode === mode));
  renderAll();
}

function renderAll() {
  renderStats();
  renderScatter();
  renderClusterTable();
  renderDetail();
}

// ---------- search ----------
function wireSearch() {
  const input = $("search-input");
  const list = $("ticker-list");
  for (const s of [...state.snap.stocks].sort((a, b) => a.t.localeCompare(b.t))) list.append(el("option", { value: s.t }, s.name));
  const resolve = (q) => {
    q = q.trim().toUpperCase();
    if (!q) return null;
    const stocks = state.snap.stocks;
    return (stocks.find((s) => s.t === q) || stocks.find((s) => s.t === q.replace(".", "-")) ||
      stocks.find((s) => s.name.toUpperCase().startsWith(q)) || stocks.find((s) => s.name.toUpperCase().includes(q)) || null);
  };
  const go = () => {
    const s = resolve(input.value);
    if (s) { selectTicker(s.t); input.setCustomValidity(""); }
    else { input.setCustomValidity("No matching ticker in the S&P 500 snapshot"); input.reportValidity(); }
  };
  input.addEventListener("change", go);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
}

// ---------- boot ----------
async function main() {
  const res = await fetch("data/snapshot.json", { cache: "no-cache" });
  state.snap = await res.json();
  const snap = state.snap;
  const ageDays = (Date.now() - Date.parse(snap.as_of + "T21:00:00Z")) / 86400000;
  $("freshness").textContent = `Prices through ${snap.as_of}. Snapshot refreshed ${snap.refreshed_at.replace("T", " ").replace("Z", " UTC")}.` + (ageDays > 5 ? " This snapshot is more than 5 days old; the last refresh may have failed." : "");
  $("freshness").dataset.asOf = snap.as_of;
  $("mode-kmeans").textContent = `K-means++ (k=${snap.kmeans.k})`;
  for (const b of document.querySelectorAll(".seg button")) b.addEventListener("click", () => setMode(b.dataset.mode));
  $("recluster-btn").addEventListener("click", recluster);
  wireScatter();
  wireSearch();
  const fromHash = decodeURIComponent(location.hash.slice(1)).toUpperCase();
  if (fromHash && snap.stocks.some((s) => s.t === fromHash)) state.selected = fromHash;
  renderAll();
  renderSmallCharts();
  renderPrediction();
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { renderScatter(); renderSmallCharts(); }, 150); });
  document.body.dataset.ready = "true";
}

main().catch((err) => {
  $("freshness").textContent = `Could not load the snapshot: ${err.message}`;
});
