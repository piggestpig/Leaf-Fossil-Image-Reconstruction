/**
 * Ruler tick scale + leaf metrics (browser port of ImageSizeDetector tick path).
 * No OCR. Expects binary mask Uint8Array length = w*h (0/1).
 */

function gaussian1d(x, sigma) {
  const n = x.length;
  const out = new Float64Array(n);
  const radius = Math.max(1, Math.ceil(sigma * 2.5));
  const kernel = new Float64Array(radius * 2 + 1);
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = v;
    ksum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = Math.min(n - 1, Math.max(0, i + k));
      s += x[j] * kernel[k + radius];
    }
    out[i] = s;
  }
  return out;
}

function correlateFullHalf(x) {
  const n = x.length;
  const corr = new Float64Array(n);
  for (let lag = 0; lag < n; lag++) {
    let s = 0;
    for (let i = 0; i < n - lag; i++) s += x[i] * x[i + lag];
    corr[lag] = s;
  }
  return corr;
}

function acfPeaks(profile, maxN = 8) {
  const L = profile.length;
  const sigma = Math.max(4, L / 40);
  let x = new Float64Array(L);
  const blur = gaussian1d(profile, sigma);
  for (let i = 0; i < L; i++) x[i] = profile[i] - blur[i];
  let mean = 0;
  for (let i = 0; i < L; i++) mean += x[i];
  mean /= L;
  let std = 0;
  for (let i = 0; i < L; i++) {
    x[i] -= mean;
    std += x[i] * x[i];
  }
  std = Math.sqrt(std / L);
  if (std < 1e-6) return [];
  for (let i = 0; i < L; i++) x[i] /= std;
  const corr = correlateFullHalf(x);
  // Always normalize; |corr[0]| can be tiny/negative with bad profiles.
  const c0 = Math.max(Math.abs(corr[0]), 1e-8);
  for (let i = 0; i < corr.length; i++) corr[i] /= c0;
  const lo = 3;
  const hi = Math.max(4, Math.floor(L / 2));
  const work = Array.from(corr.slice(lo, hi));
  const peaks = [];
  for (let n = 0; n < maxN; n++) {
    let bi = 0;
    for (let i = 1; i < work.length; i++) if (work[i] > work[bi]) bi = i;
    if (work[bi] < 0.12) break;
    let gi = lo + bi;
    if (gi > 0 && gi < corr.length - 1) {
      const y0 = corr[gi - 1];
      const y1 = corr[gi];
      const y2 = corr[gi + 1];
      const denom = 2 * (2 * y1 - y0 - y2);
      if (Math.abs(denom) > 1e-9) gi += (y0 - y2) / denom;
    }
    peaks.push([gi, work[bi]]);
    const half = Math.max(2, Math.floor(0.08 * (lo + bi)));
    const a = Math.max(0, bi - half);
    const b = Math.min(work.length, bi + half + 1);
    for (let i = a; i < b; i++) work[i] = -1;
  }
  return peaks;
}

function harmonicFundamentals(peaks) {
  if (!peaks.length) return [];
  const periods = peaks.map((p) => p[0]);
  const strengths = new Map(peaks.map(([p, s]) => [Math.round(p * 100) / 100, s]));
  const cands = [];
  for (const [T, s0] of peaks) {
    if (T < 3) continue;
    let harm = 0;
    let score = s0;
    for (const k of [2, 3, 4, 5]) {
      const target = k * T;
      let hit = false;
      for (const p of periods) {
        if (Math.abs(p - target) / target <= 0.08) {
          score += 0.55 * (strengths.get(Math.round(p * 100) / 100) || 0.3);
          harm += 1;
          hit = true;
          break;
        }
      }
      if (!hit) break;
    }
    if (harm >= 1) cands.push([T, score, harm]);
  }
  cands.sort((a, b) => b[1] - a[1] || b[2] - a[2]);
  const out = [];
  for (const [T, sc, h] of cands) {
    if (out.some(([u]) => Math.abs(T - u) / u < 0.1)) continue;
    out.push([T, sc, h]);
  }
  return out;
}

function sobelXEnergy(gray, w, h) {
  const out = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -gray[i - w - 1] +
        gray[i - w + 1] -
        2 * gray[i - 1] +
        2 * gray[i + 1] -
        gray[i + w - 1] +
        gray[i + w + 1];
      out[i] = Math.abs(gx);
    }
  }
  return out;
}

function bandTickProfiles(roiGray, w, h) {
  const t = Math.max(2, Math.floor(h * 0.34));
  const bands = [
    ["top", 0, t],
    ["bot", h - t, h],
    ["mid", t, Math.max(t + 1, h - t)],
  ];
  const out = [];
  for (const [name, y0, y1] of bands) {
    const bh = y1 - y0;
    if (bh < 2) continue;
    for (const [pol, suffix] of [
      [1, ""],
      [-1, "_inv"],
    ]) {
      const band = new Float64Array(bh * w);
      for (let y = 0; y < bh; y++) {
        for (let x = 0; x < w; x++) {
          const v = roiGray[(y0 + y) * w + x];
          band[y * w + x] = pol === 1 ? v : 255 - v;
        }
      }
      // light blur rows
      const flat = new Float64Array(bh * w);
      for (let y = 0; y < bh; y++) {
        const row = band.subarray(y * w, (y + 1) * w);
        const br = gaussian1d(row, 1.0);
        flat.set(br, y * w);
      }
      const sob = sobelXEnergy(flat, w, bh);
      const rowE = new Float64Array(bh);
      for (let y = 0; y < bh; y++) {
        let s = 0;
        for (let x = 0; x < w; x++) s += sob[y * w + x];
        rowE[y] = s / w;
      }
      let r = 0;
      for (let y = 1; y < bh; y++) if (rowE[y] > rowE[r]) r = y;
      const hh = Math.max(1, Math.floor(bh / 4));
      const r0 = Math.max(0, r - hh);
      const r1 = Math.min(bh, r + hh + 1);
      const prof = new Float64Array(w);
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let y = r0; y < r1; y++) s += sob[y * w + x];
        prof[x] = s / (r1 - r0);
      }
      out.push([name + suffix, gaussian1d(prof, 1.0)]);
    }
  }
  return out;
}

/** PCA-based min-area approximation: long/short axis + angle (deg). */
export function maskGeometry(mask, w, h) {
  let n = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      sx += x;
      sy += y;
      n += 1;
    }
  }
  if (n < 80) return null;
  const cx = sx / n;
  const cy = sy / n;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      const dx = x - cx;
      const dy = y - cy;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }
  }
  sxx /= n;
  syy /= n;
  sxy /= n;
  const trace = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const tmp = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
  const l1 = trace / 2 + tmp;
  const l2 = trace / 2 - tmp;
  let angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  // extent along eigenvectors
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      const dx = x - cx;
      const dy = y - cy;
      const u = dx * c + dy * s;
      const v = -dx * s + dy * c;
      if (u < minU) minU = u;
      if (u > maxU) maxU = u;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }
  let longSide = maxU - minU;
  let shortSide = maxV - minV;
  let angDeg = (angle * 180) / Math.PI;
  const corners = [
    [cx + c * maxU - s * maxV, cy + s * maxU + c * maxV],
    [cx + c * minU - s * maxV, cy + s * minU + c * maxV],
    [cx + c * minU - s * minV, cy + s * minU + c * minV],
    [cx + c * maxU - s * minV, cy + s * maxU + c * minV],
  ];
  if (longSide < shortSide) {
    const t = longSide;
    longSide = shortSide;
    shortSide = t;
    angDeg += 90;
  }
  return {
    cx,
    cy,
    long: longSide,
    short: Math.max(shortSide, 1e-6),
    aspect: longSide / Math.max(shortSide, 1e-6),
    area: n,
    angle: angDeg,
    corners,
    l1,
    l2,
  };
}

export function borderTouchFrac(mask, w, h, margin = 3) {
  let touch = 0;
  let area = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      area += 1;
      if (x < margin || y < margin || x >= w - margin || y >= h - margin) touch += 1;
    }
  }
  return touch / (area + 1e-6);
}

export function colorfulnessRGB(rgba, mask, w, h) {
  const rs = [];
  const gs = [];
  const bs = [];
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    rs.push(rgba[i * 4]);
    gs.push(rgba[i * 4 + 1]);
    bs.push(rgba[i * 4 + 2]);
  }
  if (rs.length < 50) return 0;
  const n = rs.length;
  const rg = new Float64Array(n);
  const yb = new Float64Array(n);
  let mrg = 0;
  let myb = 0;
  for (let i = 0; i < n; i++) {
    rg[i] = Math.abs(rs[i] - gs[i]);
    yb[i] = Math.abs(0.5 * (rs[i] + gs[i]) - bs[i]);
    mrg += rg[i];
    myb += yb[i];
  }
  mrg /= n;
  myb /= n;
  let srg = 0;
  let syb = 0;
  for (let i = 0; i < n; i++) {
    srg += (rg[i] - mrg) ** 2;
    syb += (yb[i] - myb) ** 2;
  }
  srg = Math.sqrt(srg / n);
  syb = Math.sqrt(syb / n);
  return Math.sqrt(srg * srg + syb * syb) + 0.3 * Math.sqrt(mrg * mrg + myb * myb);
}

export function straightenRoi(gray, mask, w, h, pad = 6) {
  const geo = maskGeometry(mask, w, h);
  if (!geo) return null;
  const ang = (-geo.angle * Math.PI) / 180;
  const cos = Math.cos(ang);
  const sin = Math.sin(ang);
  // rotate around center into a canvas large enough
  const diag = Math.ceil(Math.hypot(w, h)) + 2;
  const outW = diag;
  const outH = diag;
  const ox = outW / 2;
  const oy = outH / 2;
  const rotGray = new Float64Array(outW * outH);
  const rotMask = new Uint8Array(outW * outH);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const dx = x - ox;
      const dy = y - oy;
      const sx = geo.cx + dx * cos - dy * sin;
      const sy = geo.cy + dx * sin + dy * cos;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) continue;
      const fx = sx - x0;
      const fy = sy - y0;
      const i00 = y0 * w + x0;
      const g =
        gray[i00] * (1 - fx) * (1 - fy) +
        gray[i00 + 1] * fx * (1 - fy) +
        gray[i00 + w] * (1 - fx) * fy +
        gray[i00 + w + 1] * fx * fy;
      rotGray[y * outW + x] = g;
      // nearest for mask
      const mx = Math.round(sx);
      const my = Math.round(sy);
      if (mx >= 0 && my >= 0 && mx < w && my < h && mask[my * w + mx]) rotMask[y * outW + x] = 1;
    }
  }
  let minX = outW;
  let minY = outH;
  let maxX = 0;
  let maxY = 0;
  let any = false;
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      if (!rotMask[y * outW + x]) continue;
      any = true;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (!any) return null;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(outW - 1, maxX + pad);
  maxY = Math.min(outH - 1, maxY + pad);
  const rw = maxX - minX + 1;
  const rh = maxY - minY + 1;
  const roi = new Float64Array(rw * rh);
  const roiMask = new Uint8Array(rw * rh);
  const vals = [];
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      const src = (minY + y) * outW + (minX + x);
      roi[y * rw + x] = rotGray[src];
      roiMask[y * rw + x] = rotMask[src];
      if (rotMask[src]) vals.push(rotGray[src]);
    }
  }
  vals.sort((a, b) => a - b);
  const med = vals.length ? vals[Math.floor(vals.length / 2)] : 128;
  for (let i = 0; i < roi.length; i++) if (!roiMask[i]) roi[i] = med;
  return { roi, w: rw, h: rh, geo };
}

function periodScoreRoi(roi, w, h) {
  if (Math.min(w, h) < 6) return 0;
  const prof = new Float64Array(w);
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = 0; y < h; y++) s += roi[y * w + x];
    prof[x] = s / h;
  }
  const peaks = acfPeaks(prof, 3);
  if (!peaks.length) return 0;
  const [period, sig] = peaks[0];
  if (0.02 * w <= period && period <= 0.35 * w) return sig;
  return 0;
}

export function rankCandidate(mask, score, rgba, gray, w, h) {
  const geo = maskGeometry(mask, w, h);
  if (!geo) return -1e9;
  const frac = geo.area / (w * h);
  if (frac > 0.14 || frac < 0.0004 || geo.aspect < 2.05) return -1e9;
  const margin = Math.max(2, Math.floor(Math.min(h, w) / 200));
  const touch = borderTouchFrac(mask, w, h, margin);
  if (touch > 0.12 && score < 0.7) return -1e9;
  if (touch > 0.25) return -1e9;
  const cf = colorfulnessRGB(rgba, mask, w, h);
  if (cf > 40) return -1e9;
  const straightened = straightenRoi(gray, mask, w, h, 3);
  const sig = straightened ? periodScoreRoi(straightened.roi, straightened.w, straightened.h) : 0;
  return (
    6.0 * score +
    1.2 * Math.min(geo.aspect / 10.0, 1.5) +
    2.0 * sig -
    0.8 * touch * 10 -
    0.03 * cf -
    2.0 * Math.max(0.0, frac - 0.04)
  );
}

export function tickPxPerCm(roi, w, h, longPx, preferCm = true, maskAspect = null) {
  const warnings = [];
  const hyps = [];
  // Prefer 10cm only for very slender geology rulers. Photo scales (~5cm) are
  // shorter; using long/roi_h alone misfires after browser downscale to 1024.
  const aspectHW = longPx / Math.max(h, 1);
  const aspectMask = maskAspect != null ? maskAspect : aspectHW;
  const prefer10 = aspectMask >= 6.0 && aspectHW >= 3.5;

  const addHyp = (px, score, label) => {
    if (!(px >= 10 && px <= 600)) return;
    const span = longPx / px;
    if (!(span >= 3.5 && span <= 13.0)) return;
    // Clamp per-hyp score so a bad ACF cannot dominate clustering.
    score = Math.max(-5, Math.min(score, 12));
    const d5 = Math.abs(span - 5);
    const d10 = Math.abs(span - 10);
    if (prefer10) {
      if (d10 <= 2.5) score += 2.8;
      else if (d5 <= 0.8) score -= 1.2;
      else score -= 0.5;
    } else {
      // Default: photo-scale / 5cm prior (matches fossil sample set).
      if (d5 <= 1.3) score += 3.2;
      else if (d10 <= 1.2) score -= 0.8;
      else score -= 0.3;
    }
    if (preferCm) score += 0.3;
    hyps.push([px, score, label]);
  };

  for (const [tag, prof] of bandTickProfiles(roi, w, h)) {
    const peaks = acfPeaks(prof, 8);
    if (!peaks.length) continue;
    const funds = harmonicFundamentals(peaks);
    const peakPs = peaks.map((p) => p[0]);
    const hasNear = (target, tol = 0.1) =>
      peakPs.some((p) => Math.abs(p - target) / Math.max(target, 1e-6) <= tol);

    for (const [T, sc, harm] of funds) {
      if (harm >= 2 && T <= 50) {
        let mmSc = 2.8 + sc + 0.35 * harm;
        mmSc += hasNear(10 * T, 0.12) || harm >= 3 ? 1.2 : -1.0;
        addHyp(10 * T, mmSc, `${tag}:mm|T=${T.toFixed(1)}`);
      }
      if (T >= 12) {
        let cmSc = 2.4 + sc + 0.2 * harm;
        if (hasNear(2 * T, 0.1)) cmSc += 0.8;
        if (T <= 40 && harm >= 3 && hasNear(10 * T, 0.12)) cmSc -= 1.5;
        addHyp(T, cmSc, `${tag}:cm|T=${T.toFixed(1)}`);
      }
    }
    for (let i = 0; i < peakPs.length; i++) {
      for (let j = i + 1; j < peakPs.length; j++) {
        const a = peakPs[i];
        const b = peakPs[j];
        const ratio = Math.max(a, b) / (Math.min(a, b) + 1e-8);
        if (ratio >= 8.5 && ratio <= 11.5) {
          const block = Math.max(a, b);
          const fine = Math.min(a, b);
          addHyp(block, 3.2, `${tag}:dual_block|${block.toFixed(1)}`);
          addHyp(10 * fine, 3.4, `${tag}:dual_fine|${fine.toFixed(1)}`);
        }
      }
    }
    for (let i = 0; i < Math.min(3, peaks.length); i++) {
      const [p, s] = peaks[i];
      if (s >= 0.55 && p >= 20 && p <= 400) {
        addHyp(p, 1.0 + s, `${tag}:acf_cm|${p.toFixed(1)}`);
        if (p <= 50 && s >= 0.6) addHyp(10 * p, 1.2 + s, `${tag}:acf_mm|${p.toFixed(1)}`);
      }
    }
  }

  addHyp(longPx / 5, prefer10 ? 0.25 : 0.55, "bbox_5cm");
  addHyp(longPx / 10, prefer10 ? 0.55 : 0.25, "bbox_10cm");
  if (!hyps.length) return { px: 0, score: 0, labels: "", warnings: ["NO_TICK_CANDIDATE"] };

  hyps.sort((a, b) => a[0] - b[0]);
  let bestGroup = [];
  let bestSc = -1;
  for (let i = 0; i < hyps.length; i++) {
    const [vi, si] = hyps[i];
    const group = [hyps[i]];
    for (let j = i + 1; j < hyps.length; j++) {
      const [vj] = hyps[j];
      if (Math.abs(vj - vi) / Math.max(vi, 1e-6) <= 0.07) group.push(hyps[j]);
    }
    let sc = group.reduce((a, g) => a + g[1], 0);
    const bands = new Set(group.map((g) => String(g[2]).split(":")[0]));
    sc += 0.6 * Math.max(0, bands.size - 1);
    if (sc > bestSc) {
      bestSc = sc;
      bestGroup = group;
    }
  }
  let px = bestGroup.reduce((a, g) => a + g[0] * g[1], 0) / bestGroup.reduce((a, g) => a + g[1], 0);
  const half = px / 2;
  const halfGroups = hyps.filter((h) => Math.abs(h[0] - half) / Math.max(half, 1) <= 0.07);
  if (halfGroups.length) {
    const halfSc = halfGroups.reduce((a, g) => a + g[1], 0);
    const halfPx = halfGroups.reduce((a, g) => a + g[0] * g[1], 0) / halfSc;
    const target = prefer10 ? 10 : 5;
    const curSpan = longPx / px;
    const halfSpan = longPx / halfPx;
    const looksLike5 = Math.abs(curSpan - 5) <= 1.2;
    const halfWorse5 = Math.abs(halfSpan - 5) > Math.abs(curSpan - 5) + 0.5;
    const curErr = Math.abs(curSpan - target);
    const halfErr = Math.abs(halfSpan - target);
    if (!(looksLike5 && halfWorse5) && halfSc >= 0.55 * bestSc && halfErr + 0.3 < curErr) {
      px = halfPx;
      bestGroup = halfGroups;
      bestSc = halfSc;
      warnings.push("half_period_fix");
    }
  }
  const labels = [...new Set(bestGroup.map((g) => g[2]))].slice(0, 6).sort().join("+");
  return { px, score: bestSc, labels, warnings };
}

/** Leaf length/width/area from binary amodal mask + px_per_cm. */
export function leafMetricsFromMask(mask, w, h, pxPerCm) {
  const geo = maskGeometry(mask, w, h);
  const areaPx = geo ? geo.area : 0;
  const out = {
    area_px: areaPx,
    length_px: geo ? geo.long : 0,
    width_px: geo ? geo.short : 0,
    area_cm2: null,
    length_cm: null,
    width_cm: null,
    angle_deg: geo ? geo.angle : null,
    corners: geo ? geo.corners : null,
  };
  if (pxPerCm && pxPerCm > 0 && geo) {
    out.area_cm2 = areaPx / (pxPerCm * pxPerCm);
    out.length_cm = geo.long / pxPerCm;
    out.width_cm = geo.short / pxPerCm;
  }
  return out;
}

/**
 * Among ruler instances {mask, conf, cls}, pick best by tick score over top-K ranks.
 */
export function estimateScaleFromRulers(instances, rgba, gray, w, h, topK = 5) {
  const ranked = [];
  for (const inst of instances) {
    const r = rankCandidate(inst.mask, inst.conf, rgba, gray, w, h);
    if (r >= 0) ranked.push({ rank: r, ...inst });
  }
  ranked.sort((a, b) => b.rank - a.rank);
  let best = null;
  const trials = [];
  for (const cand of ranked.slice(0, topK)) {
    const st = straightenRoi(gray, cand.mask, w, h, 6);
    if (!st) continue;
    const tick = tickPxPerCm(st.roi, st.w, st.h, st.geo.long, true, st.geo.aspect);
    const span = tick.px > 0 ? st.geo.long / tick.px : 0;
    trials.push({
      cls: cand.cls,
      conf: cand.conf,
      rank: cand.rank,
      px: tick.px,
      tscore: tick.score,
      labels: tick.labels,
      span,
    });
    if (tick.px <= 0) continue;
    // Rank is only a weak prior (was exploding when ACF unnormalized).
    const rankTerm = Math.max(-2, Math.min(cand.rank, 8));
    // Prefer spans near 5cm (photo scale) or 10cm (geology ruler).
    const d5 = Math.abs(span - 5);
    const d10 = Math.abs(span - 10);
    const near = Math.min(d5, d10);
    const spanBonus = near <= 1.8 ? 5.0 - near : 0;
    const combo = tick.score + 0.15 * rankTerm + 0.5 * cand.conf + spanBonus;
    if (!best || combo > best.combo) {
      best = {
        combo,
        px_per_cm: tick.px,
        score: tick.score,
        labels: tick.labels,
        warnings: tick.warnings,
        conf: cand.conf,
        cls: cand.cls,
        rank: cand.rank,
        angle_deg: st.geo.angle,
        mask: cand.mask,
        span,
      };
    }
  }
  // Geology 10cm rulers at display-1024 often lock onto 2cm as "1cm" (span~6).
  // If a weak trial already sees span~12 (=10cm), halve the strong span~6 result.
  if (
    best &&
    best.span >= 5.5 &&
    best.span <= 7.2 &&
    trials.some((t) => t.span >= 11.2 && t.span <= 13.5 && t.px > 0)
  ) {
    best = {
      ...best,
      px_per_cm: best.px_per_cm / 2,
      span: best.span * 2,
      labels: `${best.labels}|half_for_10cm`,
    };
  }
  return { best, trials, ranked: ranked.slice(0, 6) };
}
