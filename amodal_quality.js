/**
 * No-GT amodal quality (browser port of html/tools eval / local/quality_metrics.py).
 * Confidence from sigmoid(logits); shape via largest CC + axis-aligned symmetry.
 */

const W_MEAN_IN = 0.35;
const W_MEAN_OUT = 0.25;
const W_MARGIN = 0.2;
const W_SEPARATION = 0.2;
const W_COMPACTNESS = 0.4;
const W_SMOOTHNESS = 0.3;
const W_SYMMETRY = 0.3;
const MIN_MASK_PX = 32;

function sigmoid(x) {
  if (x > 60) return 1;
  if (x < -60) return 0;
  return 1 / (1 + Math.exp(-x));
}

/** Largest connected component (4-connected) → boolean mask Uint8 0/1. */
function keepLargestCC(bin, w, h) {
  const n = w * h;
  const seen = new Uint8Array(n);
  let bestArea = 0;
  let bestStart = -1;
  const qx = new Int32Array(n);
  const qy = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (!bin[i] || seen[i]) continue;
    let area = 0;
    let head = 0;
    let tail = 0;
    qx[0] = i % w;
    qy[0] = (i / w) | 0;
    tail = 1;
    seen[i] = 1;
    while (head < tail) {
      const x = qx[head];
      const y = qy[head];
      head += 1;
      area += 1;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = ny * w + nx;
        if (seen[j] || !bin[j]) continue;
        seen[j] = 1;
        qx[tail] = nx;
        qy[tail] = ny;
        tail += 1;
      }
    }
    if (area > bestArea) {
      bestArea = area;
      bestStart = i;
    }
  }
  const out = new Uint8Array(n);
  if (bestStart < 0) return out;
  seen.fill(0);
  let head = 0;
  let tail = 0;
  qx[0] = bestStart % w;
  qy[0] = (bestStart / w) | 0;
  tail = 1;
  seen[bestStart] = 1;
  while (head < tail) {
    const x = qx[head];
    const y = qy[head];
    head += 1;
    out[y * w + x] = 1;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const j = ny * w + nx;
      if (seen[j] || !bin[j]) continue;
      seen[j] = 1;
      qx[tail] = nx;
      qy[tail] = ny;
      tail += 1;
    }
  }
  return out;
}

function confidenceMetrics(prob, mask, thr) {
  let nIn = 0;
  let nOut = 0;
  let sumIn = 0;
  let sumOut = 0;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      nIn += 1;
      sumIn += prob[i];
    } else {
      nOut += 1;
      sumOut += prob[i];
    }
  }
  const meanIn = nIn ? sumIn / nIn : NaN;
  const meanOut = nOut ? sumOut / nOut : NaN;
  const margin = nIn && nOut ? meanIn - meanOut : NaN;
  let sep = 0;
  if (nIn) sep += Math.max(0, meanIn - thr) / Math.max(1e-6, 1 - thr);
  if (nOut) sep += Math.max(0, thr - meanOut) / Math.max(1e-6, thr);
  sep *= 0.5;
  const sIn = nIn ? meanIn : 0;
  const sOut = nOut ? 1 - meanOut : 0;
  const sMargin = nIn && nOut ? Math.max(0, Math.min(1, margin)) : 0;
  const wsum = W_MEAN_IN + W_MEAN_OUT + W_MARGIN + W_SEPARATION;
  const confScore =
    (W_MEAN_IN * sIn +
      W_MEAN_OUT * sOut +
      W_MARGIN * sMargin +
      W_SEPARATION * Math.max(0, Math.min(1, sep))) /
    wsum;
  return {
    n_in: nIn,
    n_out: nOut,
    mean_prob_in: meanIn,
    mean_prob_out: meanOut,
    margin,
    separation: sep,
    conf_score: confScore,
  };
}

function shapeMetrics(mask, w, h) {
  let area = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) area += 1;
  if (area < MIN_MASK_PX) {
    return { area, compactness: NaN, smoothness: NaN, symmetry: NaN, shape_score: NaN };
  }
  // Perimeter ≈ boundary pixels (4-neigh)
  let peri = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      if (x === 0 || !mask[i - 1] || x === w - 1 || !mask[i + 1] || y === 0 || !mask[i - w] || y === h - 1 || !mask[i + w]) {
        peri += 1;
      }
    }
  }
  const compactness = Math.max(0, Math.min(1, (4 * Math.PI * area) / (peri * peri + 1e-6)));
  // Smoothness proxy: 1 / (1 + boundary fraction)
  const smoothness = 1 / (1 + (peri / Math.max(area, 1)) * 2);
  // Axis-aligned symmetry (max of H/V flip IoU)
  const flipIoU = (mode) => {
    let inter = 0;
    let union = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const a = mask[y * w + x];
        const bx = mode === "h" ? w - 1 - x : x;
        const by = mode === "v" ? h - 1 - y : y;
        const b = mask[by * w + bx];
        if (a || b) union += 1;
        if (a && b) inter += 1;
      }
    }
    return union ? inter / union : 0;
  };
  const symmetry = Math.max(flipIoU("h"), flipIoU("v"));
  const wsum = W_COMPACTNESS + W_SMOOTHNESS + W_SYMMETRY;
  const shapeScore =
    (W_COMPACTNESS * compactness + W_SMOOTHNESS * smoothness + W_SYMMETRY * symmetry) / wsum;
  return { area, compactness, smoothness, symmetry, shape_score: shapeScore };
}

/**
 * Analyze one head plane of logits (Float32Array length = size*size, or CHW slice).
 * @param {Float32Array|ArrayLike<number>} logitsPlane
 * @param {number} size  e.g. 448
 * @param {number} thr
 */
export function analyzeHead(logitsPlane, size, thr = 0.5) {
  const n = size * size;
  const prob = new Float32Array(n);
  const bin = new Uint8Array(n);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logitsPlane[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    const p = sigmoid(v);
    prob[i] = p;
    bin[i] = p >= thr ? 1 : 0;
  }
  const mask = keepLargestCC(bin, size, size);
  const conf = confidenceMetrics(prob, mask, thr);
  const shape = shapeMetrics(mask, size, size);
  let q = conf.conf_score;
  if (Number.isFinite(conf.conf_score) && Number.isFinite(shape.shape_score)) {
    q = 0.65 * conf.conf_score + 0.35 * shape.shape_score;
  }
  return { ...conf, ...shape, quality_score: q, logit_span: hi - lo };
}

/**
 * @param {Float32Array} logits  length 4*S*S
 * @param {number} size
 * @param {string[]} headNames
 * @param {Record<string, number>} thrByHead  optional per-head thr
 */
export function analyzeAmodalLogits(logits, size, headNames, thrByHead = {}) {
  const plane = size * size;
  const out = {};
  for (let hi = 0; hi < headNames.length; hi++) {
    const name = headNames[hi];
    const thr = thrByHead[name] ?? 0.5;
    const slice = logits.subarray(hi * plane, (hi + 1) * plane);
    out[name] = analyzeHead(slice, size, thr);
  }
  return out;
}

export function formatQualityBrief(quality, heads = ["amodal", "amodal_vein"], labels = {}) {
  const none = labels.none ?? "质量：未评估";
  const empty = labels.empty ?? "质量：几乎没画出来";
  const leaf = labels.leaf ?? "叶片";
  const vein = labels.vein ?? "叶脉";
  const partFn = labels.part ?? ((name, score) => `${name} ${score.toFixed(2)}`);
  const fmt = labels.fmt ?? ((parts) => `质量：${parts}`);
  if (!quality) return none;
  const parts = [];
  for (const h of heads) {
    const q = quality[h];
    if (!q || !Number.isFinite(q.quality_score)) continue;
    const name = h === "amodal" ? leaf : h === "amodal_vein" ? vein : h;
    parts.push(partFn(name, q.quality_score));
  }
  return parts.length ? fmt(parts.join(" · ")) : empty;
}
