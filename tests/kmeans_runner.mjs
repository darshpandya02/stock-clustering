// Reads {X, k, seed, nInit} JSON on stdin, prints the browser k-means result as JSON.
import { kmeans, silhouette } from "../web/kmeans.js";

let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const { X, k, seed, nInit, withSilhouette } = JSON.parse(input);
  const res = kmeans(X, k, { seed, nInit });
  const out = { labels: res.labels, inertia: res.inertia };
  if (withSilhouette) out.silhouette = silhouette(X, res.labels);
  process.stdout.write(JSON.stringify(out));
});
