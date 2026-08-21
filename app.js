/**
 * Minimal Leaf Fossil — YOLOE stone(+ruler) + Amodal Q4 (no SAM3).
 * ROI under ① · conf under ② · amodal heads + leaf size under ③.
 * Stone@WebGPU; Amodal health-check → recreate → WASM. No random Q4 smoke.
 */
import { estimateScaleFromRulers, leafMetricsFromMask } from "./ruler_scale.js";
import { analyzeAmodalLogits, formatQualityBrief } from "./amodal_quality.js";
import { t, initLang, setLang, applyStaticI18n } from "./i18n.js";

const STONE_URLS = [
  "./yoloe_stone_ruler_26s.onnx",
  "./yoloe_stone_26s.onnx",
];
const YOLOE_META_URL = "./yoloe_stone_ruler_26s.json";
/** Prefer chunked upload for GitHub (<100MB parts); fall back to monolithic .onnx. */
const AMODAL_MANIFEST = "./amodal_dino_q4.manifest.json?v=3";
const AMODAL_URL = "./amodal_dino_q4.onnx?v=3";
const ORT_URL = "./vendor/ort/ort.webgpu.min.mjs";
const SAMPLES_DIR = "./leaf_fossil_samples";

const YOLOE_SIZE = 640;
const YOLOE_PROTO = 160;
const YOLOE_MASK_DIM = 32;
const AMODAL_SIZE = 448;
const MAX_SIDE = 1024;
const OVERLAY_ALPHA = 0.5;
/** Skip DINO if stone mask area is below this (absolute or ~0.02% of image). */
const MIN_STONE_MASK_PX = 64;
/** ROI bytes: 0→0.0 erase/red, 128→0.5 neutral, 255→1.0 paint/cyan (matches retrained DINO). */
const ROI_NEUTRAL = 128;
const ROI_POS = 255;
const ROI_NEG = 0;
const HEAD_NAMES = ["visible", "amodal", "amodal_vein", "detail_vein"];
const HEAD_COLOR_DEFAULTS = {
  visible: "#00c800",
  amodal: "#c8c800",
  amodal_vein: "#0000c8",
  detail_vein: "#c800c8",
};
const HEAD_THR_IDS = [
  { range: "thrVis", num: "thrVisNum", val: "thrVisVal", en: "enVis", col: "colVis" },
  { range: "thrLeaf", num: "thrLeafNum", val: "thrLeafVal", en: "enLeaf", col: "colLeaf" },
  { range: "thrVein", num: "thrVeinNum", val: "thrVeinVal", en: "enVein", col: "colVein" },
  { range: "thrDetail", num: "thrDetailNum", val: "thrDetailVal", en: "enDetail", col: "colDetail" },
];

const $ = (id) => document.getElementById(id);
const ui = {
  badge: $("badge"),
  status: $("status"),
  btnLoad: $("btnLoad"),
  btnRun: $("btnRun"),
  btnBatch: $("btnBatch"),
  btnPrev: $("btnPrev"),
  btnNext: $("btnNext"),
  imageFile: $("imageFile"),
  cameraFile: $("cameraFile"),
  thumbStrip: $("thumbStrip"),
  albumHint: $("albumHint"),
  navInfo: $("navInfo"),
  scoreThr: $("scoreThr"),
  scoreNum: $("scoreNum"),
  scoreVal: $("scoreVal"),
  brushSize: $("brushSize"),
  brushNum: $("brushNum"),
  brushVal: $("brushVal"),
  btnDraw: $("btnDraw"),
  btnErase: $("btnErase"),
  btnRoiReset: $("btnRoiReset"),
  btnRoiClear: $("btnRoiClear"),
  cOrig: $("cOrig"),
  cRoi: $("cRoi"),
  cStone: $("cStone"),
  cAmodal: $("cAmodal"),
  imgMeta: $("imgMeta"),
  stoneMeta: $("stoneMeta"),
  amodalMeta: $("amodalMeta"),
  leafMetrics: $("leafMetrics"),
  enLeafBox: $("enLeafBox"),
  amodalQuality: $("amodalQuality"),
};

const state = {
  ort: null,
  stone: null,
  amodal: null,
  amodalBytes: null, // cached full model bytes (from chunks or single file)
  epAmodal: null,
  epStone: null,
  stoneUrl: null,
  yoloeMeta: null,
  album: [],
  index: -1,
  batchRunning: false,
  drawMode: null,
  drawing: false,
  strokeErase: false,
  scoreRefreshTimer: null,
  refreshingScore: false,
};

function current() {
  return state.index >= 0 ? state.album[state.index] : null;
}

function log(msg, kind = "") {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  const prev = ui.status.textContent ? ui.status.textContent + "\n" : "";
  ui.status.textContent = (prev + line).split("\n").slice(-50).join("\n");
  ui.status.className = "status" + (kind ? " " + kind : "");
  ui.status.scrollTop = ui.status.scrollHeight;
  console.log(line);
}

function syncThr(range, num, label, onChange) {
  const apply = (v) => {
    const x = Math.max(Number(range.min), Math.min(Number(range.max), Number(v)));
    if (!Number.isFinite(x)) return;
    range.value = String(x);
    num.value = String(x);
    label.textContent = x.toFixed(2);
    onChange?.();
  };
  range.addEventListener("input", () => apply(range.value));
  num.addEventListener("input", () => apply(num.value));
  num.addEventListener("change", () => apply(num.value));
  apply(range.value);
}

function syncBrush() {
  const apply = (v) => {
    const x = Math.max(4, Math.min(256, Math.round(Number(v))));
    if (!Number.isFinite(x)) return;
    ui.brushSize.value = String(x);
    ui.brushNum.value = String(x);
    ui.brushVal.textContent = String(x);
  };
  ui.brushSize.addEventListener("input", () => apply(ui.brushSize.value));
  ui.brushNum.addEventListener("input", () => apply(ui.brushNum.value));
  ui.brushNum.addEventListener("change", () => apply(ui.brushNum.value));
}

syncThr(ui.scoreThr, ui.scoreNum, ui.scoreVal, () => scheduleScoreRefresh());
for (const ids of HEAD_THR_IDS) {
  syncThr($(ids.range), $(ids.num), $(ids.val), () => refreshAmodalFromCache());
  $(ids.en)?.addEventListener("change", () => {
    $(ids.en).closest(".chan-row")?.classList.toggle("off", !$(ids.en).checked);
    refreshAmodalFromCache();
  });
  $(ids.col)?.addEventListener("input", () => refreshAmodalFromCache());
}
ui.enLeafBox?.addEventListener("change", () => refreshAmodalFromCache());
syncBrush();

function hexToRgb(hex) {
  const h = String(hex || "").replace("#", "").trim();
  if (h.length === 3) {
    return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)];
  }
  if (h.length >= 6) {
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  return [128, 128, 128];
}

function getAmodalHeadSettings() {
  return HEAD_THR_IDS.map((ids, i) => {
    const name = HEAD_NAMES[i];
    return {
      thr: Number($(ids.range).value),
      enabled: $(ids.en) ? $(ids.en).checked : true,
      rgb: hexToRgb($(ids.col)?.value || HEAD_COLOR_DEFAULTS[name]),
    };
  });
}

function qualityLabels() {
  return {
    none: t("qualityNoneDyn"),
    empty: t("qualityEmpty"),
    leaf: t("qualityLeaf"),
    vein: t("qualityVein"),
    part: (name, score) => t("qualityPart", { name, score: score.toFixed(2) }),
    fmt: (parts) => t("qualityFmt", { parts }),
  };
}

function refreshLangUi() {
  applyStaticI18n();
  if (ui.btnLoad.disabled && /加载中|Loading/.test(ui.btnLoad.textContent || "")) {
    ui.btnLoad.textContent = t("btnLoading");
  } else if (state.stone && state.amodal) {
    ui.btnLoad.textContent = t("btnReload");
  } else {
    ui.btnLoad.textContent = t("btnLoad");
  }
  updateReady();
  renderThumbs();
  const item = current();
  if (item) {
    updateItemMetas(item);
    formatLeafMetrics(item);
    formatAmodalQuality(item);
  }
}

function updateItemMetas(item) {
  if (!item) return;
  if (item.w && item.h) ui.imgMeta.textContent = `${item.w}×${item.h}`;
  if (item.stoneCanvas) {
    ui.stoneMeta.textContent = item.topScore != null
      ? t("metaStoneScore", { score: item.topScore.toFixed(3) })
      : "";
  } else {
    ui.stoneMeta.textContent = t("metaUntreated");
  }
  if (item.amodalCanvas) {
    ui.amodalMeta.textContent = item.stoneEmpty ? t("metaNoStone") : t("metaLayers");
  } else {
    ui.amodalMeta.textContent = t("metaUntreated");
  }
}

function updateReady() {
  const hasAlbum = state.album.length > 0;
  const models = !!(state.stone && state.amodal);
  const item = current();
  const busy = state.batchRunning || state.refreshingScore;
  ui.btnRun.disabled = !(models && item && !busy);
  ui.btnBatch.disabled = !(models && hasAlbum && !busy);
  ui.btnPrev.disabled = state.index <= 0 || busy;
  ui.btnNext.disabled = state.index < 0 || state.index >= state.album.length - 1 || busy;
  ui.cRoi.classList.toggle("disabled", !item);
  ui.btnDraw.classList.toggle("active", state.drawMode === "draw");
  ui.btnErase.classList.toggle("active", state.drawMode === "erase");
}

async function importOrt() {
  const abs = new URL(`${ORT_URL}?v=1`, window.location.href).href;
  log(`导入 ORT: ${abs}`);
  const res = await fetch(abs, { cache: "no-store" });
  if (!res.ok) throw new Error(`ORT HTTP ${res.status}`);
  let code = await res.text();
  const realUrl = abs.split("?")[0];
  code = code.split("import.meta.url").join(JSON.stringify(realUrl));
  const blobUrl = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
  const mod = await import(blobUrl);
  const ort = mod.default ?? mod;
  if (!ort?.InferenceSession) throw new Error("ORT 模块无 InferenceSession");
  ort.env.wasm.wasmPaths = new URL("./", realUrl).href;
  ort.env.wasm.simd = true;

  // Multi-thread WASM needs crossOriginIsolated (COOP+COEP). Else keep 1.
  const cores = navigator.hardwareConcurrency || 4;
  if (globalThis.crossOriginIsolated) {
    ort.env.wasm.numThreads = Math.min(4, Math.max(2, Math.floor(cores / 2)));
  } else {
    ort.env.wasm.numThreads = 1;
  }

  // Prefer discrete / high-performance GPU (helps dual-GPU laptops; also sets intent on Apple Silicon).
  if (ort.env.webgpu) {
    try {
      ort.env.webgpu.powerPreference = "high-performance";
    } catch { /* older ORT */ }
  }
  if (navigator.gpu?.requestAdapter) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) {
        const info = adapter.info || (await adapter.requestAdapterInfo?.().catch(() => null));
        const device = await adapter.requestDevice();
        ort.env.webgpu.adapter = adapter;
        ort.env.webgpu.device = device;
        const label = info
          ? `${info.vendor || ""} ${info.architecture || info.device || ""}`.trim()
          : "(adapter ok)";
        log(`WebGPU high-performance · ${label || "ok"}`, "ok");
      } else {
        log("WebGPU requestAdapter 返回空（将主要依赖 WASM）", "err");
      }
    } catch (e) {
      log(`WebGPU 设备申请失败: ${e.message || e}（仍可试 EP）`, "err");
    }
  } else {
    log("无 navigator.gpu — 仅 WASM", "err");
  }

  log(
    `ORT 就绪 · wasmThreads=${ort.env.wasm.numThreads} · ` +
      `cores=${cores} · crossOriginIsolated=${!!globalThis.crossOriginIsolated} · ` +
      `gpu=${!!navigator.gpu}`,
    "ok",
  );
  state.ort = ort;
  return ort;
}

async function createSession(source, label, prefer = ["webgpu"]) {
  const ort = state.ort || (await importOrt());
  let lastErr;
  for (const ep of prefer) {
    log(`创建 Session (${ep}) · ${label}`);
    const t0 = performance.now();
    try {
      const session = await ort.InferenceSession.create(source, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      log(`Session OK · ${((performance.now() - t0) / 1000).toFixed(1)}s · ep=${ep} · in=[${session.inputNames}]`, "ok");
      return { session, ep };
    } catch (e) {
      lastErr = e;
      log(`  ${ep} 失败: ${e.message || e}`, "err");
    }
  }
  throw lastErr || new Error(`无法创建 Session · ${label}`);
}

/** Load model bytes: prefer monolithic .onnx (local), else GitHub chunk manifest. */
async function loadModelBytes(manifestUrl, singleUrl, label) {
  if (await urlExists(singleUrl)) {
    log(`加载 ${label}：${singleUrl}`);
    return await (await fetch(singleUrl, { cache: "no-store" })).arrayBuffer();
  }
  if (await urlExists(manifestUrl)) {
    const man = await (await fetch(manifestUrl, { cache: "no-store" })).json();
    const parts = man.chunks || [];
    if (!parts.length) throw new Error(`${manifestUrl} has no chunks`);
    log(`加载 ${label}：${parts.length} 个分片（共 ${((man.total_bytes || 0) / (1024 * 1024)).toFixed(1)} MB）…`);
    const buffers = [];
    let total = 0;
    for (const c of parts) {
      const name = typeof c === "string" ? c : c.file;
      const url = name.includes("/") ? name : `./${name}`;
      log(`  ↓ ${url}`);
      const buf = await (await fetch(url, { cache: "no-store" })).arrayBuffer();
      buffers.push(new Uint8Array(buf));
      total += buf.byteLength;
    }
    if (man.total_bytes && total !== man.total_bytes) {
      throw new Error(`${label} 分片总长 ${total} ≠ manifest ${man.total_bytes}`);
    }
    const out = new Uint8Array(total);
    let off = 0;
    for (const b of buffers) {
      out.set(b, off);
      off += b.length;
    }
    log(`  拼接完成 ${(total / (1024 * 1024)).toFixed(1)} MB`, "ok");
    return out.buffer;
  }
  throw new Error(`未找到 ${label}（需要 ${singleUrl} 或 ${manifestUrl} 分片）`);
}

async function recreateAmodalSession(prefer) {
  try { await state.amodal?.release?.(); } catch { /* ignore */ }
  if (!state.amodalBytes) {
    state.amodalBytes = await loadModelBytes(AMODAL_MANIFEST, AMODAL_URL, "Amodal Q4");
  }
  const { session, ep } = await createSession(
    state.amodalBytes,
    "amodal_dino_q4 [recreate]",
    prefer,
  );
  state.amodal = session;
  state.epAmodal = ep;
  ui.badge.textContent =
    `YOLOE+Q4 · S:${(state.epStone || "?").toUpperCase()} · A:${ep.toUpperCase()}`;
  ui.badge.classList.add("ok");
  return session;
}

function amodalLogitsLookBad(logits) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (!Number.isFinite(v)) return { bad: true, reason: "non-finite", min, max, span: 0, visFrac: 0 };
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min;
  const plane = AMODAL_SIZE * AMODAL_SIZE;
  let vis = 0;
  for (let p = 0; p < plane; p++) {
    if (sigmoid(logits[p]) > 0.5) vis += 1;
  }
  const visFrac = vis / plane;
  // Retained model: healthy visible often 20–35%. Old vis>22% forced WASM every run.
  if (span < 8) return { bad: true, reason: `span=${span.toFixed(1)}<8`, min, max, span, visFrac };
  if (visFrac > 0.85) return { bad: true, reason: `visible=${(visFrac * 100).toFixed(1)}%>85%`, min, max, span, visFrac };
  if (span < 25 && visFrac > 0.40) {
    return { bad: true, reason: `poison span=${span.toFixed(1)}&vis=${(visFrac * 100).toFixed(1)}%`, min, max, span, visFrac };
  }
  return { bad: false, min, max, span, visFrac };
}

function tensorToFloat32(data) {
  if (data instanceof Float32Array) return data;
  if (typeof Float16Array !== "undefined" && data instanceof Float16Array) {
    return Float32Array.from(data);
  }
  return Float32Array.from(data);
}

function sigmoid(x) {
  if (x >= 0) { const z = Math.exp(-x); return 1 / (1 + z); }
  const z = Math.exp(x); return z / (1 + z);
}

async function limitBitmapSide(bitmap, maxSide) {
  const s = Math.max(bitmap.width, bitmap.height);
  if (s <= maxSide) return bitmap;
  const scale = maxSide / s;
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(bitmap, 0, 0, w, h);
  const scaled = await createImageBitmap(c);
  if (bitmap.close) bitmap.close();
  return scaled;
}

function showCanvas(canvasEl, source) {
  const ctx = canvasEl.getContext("2d");
  if (!source) {
    canvasEl.width = canvasEl.height = 0;
    return;
  }
  canvasEl.width = source.width;
  canvasEl.height = source.height;
  ctx.drawImage(source, 0, 0);
}

function letterboxYoloe(bitmap) {
  const size = YOLOE_SIZE;
  const iw = bitmap.width;
  const ih = bitmap.height;
  const r = Math.min(size / iw, size / ih);
  const nw = Math.round(iw * r);
  const nh = Math.round(ih * r);
  const padL = Math.floor((size - nw) / 2);
  const padT = Math.floor((size - nh) / 2);
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, padL, padT, nw, nh);
  const { data } = ctx.getImageData(0, 0, size, size);
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = data[i * 4] / 255;
    out[n + i] = data[i * 4 + 1] / 255;
    out[2 * n + i] = data[i * 4 + 2] / 255;
  }
  return { chw: out, padL, padT, nw, nh, iw, ih };
}

function upsampleMask2d(src, mh, mw, outH, outW) {
  const dst = new Float32Array(outH * outW);
  const yScale = (mh - 1) / Math.max(outH - 1, 1);
  const xScale = (mw - 1) / Math.max(outW - 1, 1);
  for (let y = 0; y < outH; y++) {
    const fy = y * yScale;
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, mh - 1);
    const wy = fy - y0;
    for (let x = 0; x < outW; x++) {
      const fx = x * xScale;
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, mw - 1);
      const wx = fx - x0;
      const v00 = src[y0 * mw + x0];
      const v01 = src[y0 * mw + x1];
      const v10 = src[y1 * mw + x0];
      const v11 = src[y1 * mw + x1];
      dst[y * outW + x] =
        v00 * (1 - wy) * (1 - wx) +
        v01 * (1 - wy) * wx +
        v10 * wy * (1 - wx) +
        v11 * wy * wx;
    }
  }
  return dst;
}

function yoloeProtoMaskArea(coeffs, proto, x1, y1, x2, y2) {
  const mh = YOLOE_PROTO;
  const mw = YOLOE_PROTO;
  const size = YOLOE_SIZE;
  const sx0 = Math.max(0, Math.floor((x1 / size) * mw));
  const sx1 = Math.min(mw, Math.ceil((x2 / size) * mw));
  const sy0 = Math.max(0, Math.floor((y1 / size) * mh));
  const sy1 = Math.min(mh, Math.ceil((y2 / size) * mh));
  let area = 0;
  for (let y = sy0; y < sy1; y++) {
    for (let x = sx0; x < sx1; x++) {
      const p = y * mw + x;
      let s = 0;
      for (let k = 0; k < YOLOE_MASK_DIM; k++) s += coeffs[k] * proto[k * mh * mw + p];
      if (s > 0) area += 1;
    }
  }
  return area;
}

function keepLargestComponent(soft, w, h) {
  const n = w * h;
  const seen = new Uint8Array(n);
  const qx = new Int32Array(n);
  const qy = new Int32Array(n);
  let bestArea = 0;
  let bestSx = -1;
  let bestSy = -1;
  const nbr = [-1, -1, 0, -1, 1, -1, -1, 0, 1, 0, -1, 1, 0, 1, 1, 1];
  for (let i = 0; i < n; i++) {
    if (soft[i] <= 0.5 || seen[i]) continue;
    const sx = i % w;
    const sy = (i / w) | 0;
    let area = 0;
    let head = 0;
    let tail = 0;
    qx[tail] = sx;
    qy[tail] = sy;
    tail += 1;
    seen[i] = 1;
    while (head < tail) {
      const x = qx[head];
      const y = qy[head];
      head += 1;
      area += 1;
      for (let k = 0; k < 16; k += 2) {
        const nx = x + nbr[k];
        const ny = y + nbr[k + 1];
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = ny * w + nx;
        if (seen[j] || soft[j] <= 0.5) continue;
        seen[j] = 1;
        qx[tail] = nx;
        qy[tail] = ny;
        tail += 1;
      }
    }
    if (area > bestArea) {
      bestArea = area;
      bestSx = sx;
      bestSy = sy;
    }
  }
  if (bestSx < 0) return soft;
  const keep = new Float32Array(n);
  seen.fill(0);
  let head = 0;
  let tail = 0;
  qx[tail] = bestSx;
  qy[tail] = bestSy;
  tail += 1;
  seen[bestSy * w + bestSx] = 1;
  while (head < tail) {
    const x = qx[head];
    const y = qy[head];
    head += 1;
    keep[y * w + x] = 1;
    for (let k = 0; k < 16; k += 2) {
      const nx = x + nbr[k];
      const ny = y + nbr[k + 1];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const j = ny * w + nx;
      if (seen[j] || soft[j] <= 0.5) continue;
      seen[j] = 1;
      qx[tail] = nx;
      qy[tail] = ny;
      tail += 1;
    }
  }
  return keep;
}

/** Decode one YOLOE det row → binary mask at original size (0/1 Float32Array). */
function yoloeRowToSoft(det, proto, meta, rowIndex) {
  const { padL, padT, nw, nh, iw, ih } = meta;
  const base = rowIndex * 38;
  const x1 = det[base];
  const y1 = det[base + 1];
  const x2 = det[base + 2];
  const y2 = det[base + 3];
  const coeffs = det.subarray(base + 6, base + 38);
  const mh = YOLOE_PROTO;
  const mw = YOLOE_PROTO;
  const logits = new Float32Array(mh * mw);
  for (let p = 0; p < mh * mw; p++) {
    let s = 0;
    for (let k = 0; k < YOLOE_MASK_DIM; k++) s += coeffs[k] * proto[k * mh * mw + p];
    logits[p] = s;
  }
  const size = YOLOE_SIZE;
  const up = upsampleMask2d(logits, mh, mw, size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (x < x1 || x >= x2 || y < y1 || y >= y2) up[y * size + x] = -1e9;
    }
  }
  const soft = new Float32Array(iw * ih);
  for (let y = 0; y < ih; y++) {
    const sy = (y * nh) / ih + padT;
    const y0 = Math.floor(sy);
    const y1i = Math.min(y0 + 1, size - 1);
    const wy = sy - y0;
    for (let x = 0; x < iw; x++) {
      const sx = (x * nw) / iw + padL;
      const x0 = Math.floor(sx);
      const x1i = Math.min(x0 + 1, size - 1);
      const wx = sx - x0;
      const v00 = up[y0 * size + x0];
      const v01 = up[y0 * size + x1i];
      const v10 = up[y1i * size + x0];
      const v11 = up[y1i * size + x1i];
      soft[y * iw + x] =
        v00 * (1 - wy) * (1 - wx) +
        v01 * (1 - wy) * wx +
        v10 * wy * (1 - wx) +
        v11 * wy * wx;
    }
  }
  for (let i = 0; i < soft.length; i++) soft[i] = soft[i] > 0 ? 1 : 0;
  return soft;
}

/** Among conf>thr (optional cls filter) pick largest proto-mask, then largest CC. */
function decodeYoloeMask(det, proto, meta, confThr, allowedCls = null) {
  const nDet = det.length / 38;
  const { iw, ih } = meta;
  let bestI = -1;
  let bestArea = -1;
  let bestConf = 0;
  for (let i = 0; i < nDet; i++) {
    const base = i * 38;
    const conf = det[base + 4];
    if (conf <= confThr) continue;
    const cls = Math.round(det[base + 5]);
    if (allowedCls && !allowedCls.has(cls)) continue;
    const area = yoloeProtoMaskArea(
      det.subarray(base + 6, base + 38),
      proto,
      det[base],
      det[base + 1],
      det[base + 2],
      det[base + 3],
    );
    if (area > bestArea) {
      bestArea = area;
      bestI = i;
      bestConf = conf;
    }
  }
  if (bestI < 0) return { soft: new Float32Array(iw * ih), topScore: 0 };
  const soft = yoloeRowToSoft(det, proto, meta, bestI);
  return { soft: keepLargestComponent(soft, iw, ih), topScore: bestConf };
}

/** All YOLOE instances for given class ids (Uint8 mask 0/1). */
function decodeYoloeClassInstances(det, proto, meta, confThr, classIds) {
  const nDet = det.length / 38;
  const { iw, ih } = meta;
  const allow = classIds instanceof Set ? classIds : new Set(classIds || []);
  const out = [];
  for (let i = 0; i < nDet; i++) {
    const base = i * 38;
    const conf = det[base + 4];
    if (conf <= confThr) continue;
    const cls = Math.round(det[base + 5]);
    if (allow.size && !allow.has(cls)) continue;
    const soft = yoloeRowToSoft(det, proto, meta, i);
    let area = 0;
    const mask = new Uint8Array(iw * ih);
    for (let p = 0; p < soft.length; p++) {
      if (soft[p] > 0.5) {
        mask[p] = 1;
        area += 1;
      }
    }
    if (area < 30) continue;
    out.push({ cls, conf, mask, area });
  }
  return out;
}

function binarySoftToCanvas(soft, outW, outH) {
  const c = document.createElement("canvas");
  c.width = outW;
  c.height = outH;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const img = ctx.createImageData(outW, outH);
  for (let p = 0; p < soft.length; p++) {
    const on = soft[p] > 0.5 ? 255 : 0;
    img.data[p * 4] = 255;
    img.data[p * 4 + 1] = 255;
    img.data[p * 4 + 2] = 255;
    img.data[p * 4 + 3] = on;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function makeWhiteBgStone(bitmap, maskCanvas) {
  const w = bitmap.width, h = bitmap.height;
  const out = document.createElement("canvas");
  out.width = w; out.height = h;
  const ctx = out.getContext("2d", { willReadFrequently: true });
  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext("2d", { willReadFrequently: true });
  tctx.drawImage(bitmap, 0, 0);
  tctx.globalCompositeOperation = "destination-in";
  tctx.drawImage(maskCanvas, 0, 0);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(tmp, 0, 0);
  return out;
}

function makeSolidWhiteCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  return c;
}

function isStoneMaskEmpty(soft, w, h) {
  if (!soft?.length) return true;
  let area = 0;
  for (let i = 0; i < soft.length; i++) if (soft[i] > 0.5) area += 1;
  const minPx = Math.max(MIN_STONE_MASK_PX, Math.floor(w * h * 0.0002));
  return area < minPx;
}

function applyNoStoneSkipAmodal(item) {
  item.stoneEmpty = true;
  item.amodalLogits = null;
  item.amodalCanvas = makeSolidWhiteCanvas(item.w, item.h);
  item.leafMetrics = null;
  item.amodalQuality = null;
  if (!item.stoneCanvas) item.stoneCanvas = makeSolidWhiteCanvas(item.w, item.h);
  if (ui.leafMetrics) ui.leafMetrics.textContent = t("sizeSkipped");
  if (ui.amodalQuality) ui.amodalQuality.textContent = t("qualitySkipped");
}

function resizeCanvas(src, size) {
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  c.getContext("2d").drawImage(src, 0, 0, size, size);
  return c;
}

function ensureRoi(item) {
  const n = item.w * item.h;
  if (!item.roi || item.roi.length !== n) {
    item.roi = new Uint8ClampedArray(n);
    item.roi.fill(ROI_NEUTRAL);
    item.roiEdited = false;
  }
  return item.roi;
}

function paintRoiOverlay(item) {
  const c = ui.cRoi;
  c.width = item.w;
  c.height = item.h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!item.roiEdited) {
    ctx.clearRect(0, 0, c.width, c.height);
    return;
  }
  const img = ctx.createImageData(item.w, item.h);
  const roi = ensureRoi(item);
  for (let i = 0; i < roi.length; i++) {
    const o = i * 4;
    const v = roi[i];
    if (v >= 200) {
      img.data[o] = 0; img.data[o + 1] = 210; img.data[o + 2] = 230; img.data[o + 3] = 90;
    } else if (v <= 40) {
      img.data[o] = 220; img.data[o + 1] = 40; img.data[o + 2] = 40; img.data[o + 3] = 110;
    } else {
      img.data[o + 3] = 0;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function canvasToClientPos(canvas, ev) {
  const rect = canvas.getBoundingClientRect();
  const x = ((ev.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((ev.clientY - rect.top) / rect.height) * canvas.height;
  return { x, y };
}

function brushAt(item, x, y, erase) {
  const roi = ensureRoi(item);
  item.roiEdited = true;
  const r = Number(ui.brushSize.value);
  const r2 = r * r;
  const x0 = Math.max(0, Math.floor(x - r));
  const y0 = Math.max(0, Math.floor(y - r));
  const x1 = Math.min(item.w - 1, Math.ceil(x + r));
  const y1 = Math.min(item.h - 1, Math.ceil(y + r));
  const val = erase ? ROI_NEG : ROI_POS;
  for (let yy = y0; yy <= y1; yy++) {
    for (let xx = x0; xx <= x1; xx++) {
      const dx = xx + 0.5 - x, dy = yy + 0.5 - y;
      if (dx * dx + dy * dy <= r2) roi[yy * item.w + xx] = val;
    }
  }
}

function bindRoiDrawing() {
  const c = ui.cRoi;
  const onDown = (ev) => {
    const item = current();
    if (!item) return;
    if (ev.button === 2) {
      state.drawMode = "erase";
      updateReady();
    } else if (ev.button === 0) {
      if (!state.drawMode) {
        state.drawMode = "draw";
        updateReady();
      }
    } else return;
    const erase = ev.button === 2 || state.drawMode === "erase";
    state.drawing = true;
    state.strokeErase = erase;
    try { c.setPointerCapture(ev.pointerId); } catch { /* ignore */ }
    const p = canvasToClientPos(c, ev);
    brushAt(item, p.x, p.y, erase);
    paintRoiOverlay(item);
    ev.preventDefault();
  };
  const onMove = (ev) => {
    if (!state.drawing) return;
    const item = current();
    if (!item) return;
    const p = canvasToClientPos(c, ev);
    brushAt(item, p.x, p.y, !!state.strokeErase);
    paintRoiOverlay(item);
  };
  const onUp = (ev) => {
    state.drawing = false;
    state.strokeErase = false;
    try { if (ev?.pointerId != null) c.releasePointerCapture(ev.pointerId); } catch { /* ignore */ }
  };
  c.addEventListener("pointerdown", onDown);
  c.addEventListener("pointermove", onMove);
  c.addEventListener("pointerup", onUp);
  c.addEventListener("pointercancel", onUp);
  c.addEventListener("contextmenu", (ev) => ev.preventDefault());
  window.addEventListener("pointerup", onUp);
}

function buildAmodalInput(stoneCanvas, roi, srcW, srcH) {
  // RGB bilinear via canvas; ROI nearest → keep ternary 0 / 0.5 / 1.0
  const stoneSmall = resizeCanvas(stoneCanvas, AMODAL_SIZE);
  const sdata = stoneSmall.getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, AMODAL_SIZE, AMODAL_SIZE).data;
  const n = AMODAL_SIZE * AMODAL_SIZE;
  const out = new Float32Array(4 * n);
  const scaleX = srcW / AMODAL_SIZE;
  const scaleY = srcH / AMODAL_SIZE;
  for (let y = 0; y < AMODAL_SIZE; y++) {
    const sy = Math.min(srcH - 1, Math.floor((y + 0.5) * scaleY));
    for (let x = 0; x < AMODAL_SIZE; x++) {
      const sx = Math.min(srcW - 1, Math.floor((x + 0.5) * scaleX));
      const i = y * AMODAL_SIZE + x;
      out[i] = sdata[i * 4] / 255;
      out[n + i] = sdata[i * 4 + 1] / 255;
      out[2 * n + i] = sdata[i * 4 + 2] / 255;
      out[3 * n + i] = roi[sy * srcW + sx] / 255;
    }
  }
  return out;
}

function overlayAmodalOnStone(stoneCanvas, logits, settings) {
  const plane = AMODAL_SIZE * AMODAL_SIZE;
  const outW = stoneCanvas.width;
  const outH = stoneCanvas.height;
  const counts = [];
  const out = document.createElement("canvas");
  out.width = outW;
  out.height = outH;
  const octx = out.getContext("2d");
  octx.drawImage(stoneCanvas, 0, 0);
  for (let hi = 0; hi < HEAD_NAMES.length; hi++) {
    const cfg = settings[hi] || { thr: 0.5, enabled: true, rgb: [128, 128, 128] };
    const thr = cfg.thr ?? 0.5;
    let onCount = 0;
    for (let p = 0; p < plane; p++) {
      if (sigmoid(logits[hi * plane + p]) > thr) onCount += 1;
    }
    counts.push(onCount);
    if (!cfg.enabled) continue;
    const [cr, cg, cb] = cfg.rgb;
    const low = document.createElement("canvas");
    low.width = AMODAL_SIZE;
    low.height = AMODAL_SIZE;
    const lctx = low.getContext("2d", { willReadFrequently: true });
    const img = lctx.createImageData(AMODAL_SIZE, AMODAL_SIZE);
    for (let p = 0; p < plane; p++) {
      const on = sigmoid(logits[hi * plane + p]) > thr;
      img.data[p * 4] = cr;
      img.data[p * 4 + 1] = cg;
      img.data[p * 4 + 2] = cb;
      img.data[p * 4 + 3] = on ? Math.round(255 * OVERLAY_ALPHA) : 0;
    }
    lctx.putImageData(img, 0, 0);
    octx.drawImage(low, 0, 0, outW, outH);
  }
  return { canvas: out, counts };
}

/** Upsample amodal head logits → binary Uint8 mask at original size. */
function amodalHeadMaskOrig(logits, headIndex, thr, outW, outH) {
  const plane = AMODAL_SIZE * AMODAL_SIZE;
  const mask = new Uint8Array(outW * outH);
  const base = headIndex * plane;
  for (let y = 0; y < outH; y++) {
    const sy = ((y + 0.5) * AMODAL_SIZE) / outH - 0.5;
    const y0 = Math.max(0, Math.min(AMODAL_SIZE - 1, Math.floor(sy)));
    const y1 = Math.min(AMODAL_SIZE - 1, y0 + 1);
    const wy = Math.max(0, Math.min(1, sy - y0));
    for (let x = 0; x < outW; x++) {
      const sx = ((x + 0.5) * AMODAL_SIZE) / outW - 0.5;
      const x0 = Math.max(0, Math.min(AMODAL_SIZE - 1, Math.floor(sx)));
      const x1 = Math.min(AMODAL_SIZE - 1, x0 + 1);
      const wx = Math.max(0, Math.min(1, sx - x0));
      const v00 = sigmoid(logits[base + y0 * AMODAL_SIZE + x0]);
      const v01 = sigmoid(logits[base + y0 * AMODAL_SIZE + x1]);
      const v10 = sigmoid(logits[base + y1 * AMODAL_SIZE + x0]);
      const v11 = sigmoid(logits[base + y1 * AMODAL_SIZE + x1]);
      const v =
        v00 * (1 - wy) * (1 - wx) +
        v01 * (1 - wy) * wx +
        v10 * wy * (1 - wx) +
        v11 * wy * wx;
      mask[y * outW + x] = v > thr ? 1 : 0;
    }
  }
  return mask;
}

function formatLeafMetrics(item) {
  if (!ui.leafMetrics) return;
  const m = item?.leafMetrics;
  const s = item?.scale;
  if (!m) {
    ui.leafMetrics.textContent = t("sizePending");
    return;
  }
  const parts = [];
  if (s?.px_per_cm) {
    parts.push(t("sizeScale", { ppc: s.px_per_cm.toFixed(2) }));
    if (s.clsName) parts.push(s.clsName);
  } else {
    parts.push(t("sizeNoRuler"));
  }
  if (m.length_cm != null) {
    parts.push(t("sizeCm", {
      l: m.length_cm.toFixed(2),
      w: m.width_cm.toFixed(2),
      a: m.area_cm2.toFixed(2),
    }));
  } else {
    parts.push(t("sizePx", {
      l: m.length_px.toFixed(0),
      w: m.width_px.toFixed(0),
      a: m.area_px,
    }));
  }
  ui.leafMetrics.textContent = t("sizeLabel", { body: parts.join(" · ") });
}

function formatAmodalQuality(item) {
  if (!ui.amodalQuality) return;
  ui.amodalQuality.textContent = formatQualityBrief(item?.amodalQuality, ["amodal", "amodal_vein"], qualityLabels());
}

function updateAmodalQuality(item) {
  if (!item?.amodalLogits) {
    item.amodalQuality = null;
    formatAmodalQuality(item);
    return;
  }
  const settings = getAmodalHeadSettings();
  const thrByHead = {};
  for (let i = 0; i < HEAD_NAMES.length; i++) {
    thrByHead[HEAD_NAMES[i]] = settings[i]?.thr ?? 0.5;
  }
  item.amodalQuality = analyzeAmodalLogits(
    item.amodalLogits,
    AMODAL_SIZE,
    HEAD_NAMES,
    thrByHead,
  );
  formatAmodalQuality(item);
}

function drawLeafBoxOnCanvas(canvas, item) {
  if (!ui.enLeafBox?.checked) return;
  const m = item?.leafMetrics;
  if (!m?.corners || m.corners.length < 4) return;
  const ctx = canvas.getContext("2d");
  const pts = m.corners;
  ctx.save();
  ctx.strokeStyle = "#ffcc33";
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 280));
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.stroke();

  let label;
  if (m.length_cm != null) {
    label = `L ${m.length_cm.toFixed(2)} cm\nW ${m.width_cm.toFixed(2)} cm\nA ${m.area_cm2.toFixed(2)} cm²`;
  } else {
    label = `L ${m.length_px.toFixed(0)} px\nW ${m.width_px.toFixed(0)} px\nA ${m.area_px} px²`;
  }
  const fontPx = Math.max(12, Math.round(Math.min(canvas.width, canvas.height) / 42));
  ctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = "top";
  const lines = label.split("\n");
  const pad = Math.round(fontPx * 0.35);
  const lineH = Math.round(fontPx * 1.25);
  let tw = 0;
  for (const line of lines) tw = Math.max(tw, ctx.measureText(line).width);
  const boxW = tw + pad * 2;
  const boxH = lineH * lines.length + pad * 2;
  const margin = Math.max(6, Math.round(fontPx * 0.4));
  const lx = margin;
  const ly = canvas.height - boxH - margin;
  ctx.fillRect(lx, ly, boxW, boxH);
  ctx.strokeStyle = "#ffcc33";
  ctx.lineWidth = 1;
  ctx.strokeRect(lx, ly, boxW, boxH);
  ctx.fillStyle = "#ffe082";
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], lx + pad, ly + pad + i * lineH);
  }
  ctx.restore();
}

function paintAmodalWithMetrics(item) {
  if (!item?.amodalLogits || !item.stoneCanvas) return null;
  const settings = getAmodalHeadSettings();
  const { canvas, counts } = overlayAmodalOnStone(item.stoneCanvas, item.amodalLogits, settings);
  drawLeafBoxOnCanvas(canvas, item);
  item.amodalCanvas = canvas;
  return counts;
}

function updateLeafMetrics(item) {
  if (!item?.amodalLogits || !item.bitmap) return;
  const settings = getAmodalHeadSettings();
  const leafThr = settings[1]?.thr ?? 0.5;
  const leafMask = amodalHeadMaskOrig(item.amodalLogits, 1, leafThr, item.w, item.h);

  let pxPerCm = item.scale?.px_per_cm || 0;
  if (!item.scale && item.rulerInstances?.length) {
    const c = document.createElement("canvas");
    c.width = item.w;
    c.height = item.h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(item.bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, item.w, item.h).data;
    const gray = new Float64Array(item.w * item.h);
    for (let i = 0; i < gray.length; i++) {
      gray[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
    }
    const { best } = estimateScaleFromRulers(
      item.rulerInstances,
      rgba,
      gray,
      item.w,
      item.h,
      5,
    );
    if (best) {
      const names = state.yoloeMeta?.classes || [];
      item.scale = {
        px_per_cm: best.px_per_cm,
        conf: best.conf,
        cls: best.cls,
        clsName: names[best.cls] || String(best.cls),
        labels: best.labels,
        rank: best.rank,
        angle_deg: best.angle_deg,
      };
      pxPerCm = best.px_per_cm;
      log(t("logScaleOk"), "ok");
    } else {
      item.scale = { px_per_cm: 0, error: "NO_SCALE" };
      log(t("logScaleMiss"), "err");
    }
  }

  item.leafMetrics = leafMetricsFromMask(leafMask, item.w, item.h, pxPerCm);
  formatLeafMetrics(item);
}

function refreshAmodalFromCache() {
  const item = current();
  if (!item?.amodalLogits || !item.stoneCanvas) return;
  const settings = getAmodalHeadSettings();
  if (item.scale?.px_per_cm) {
    const leafThr = settings[1]?.thr ?? 0.5;
    const leafMask = amodalHeadMaskOrig(item.amodalLogits, 1, leafThr, item.w, item.h);
    item.leafMetrics = leafMetricsFromMask(leafMask, item.w, item.h, item.scale.px_per_cm);
    formatLeafMetrics(item);
  } else if (item.rulerInstances?.length) {
    updateLeafMetrics(item);
  } else if (item.leafMetrics) {
    const leafThr = settings[1]?.thr ?? 0.5;
    const leafMask = amodalHeadMaskOrig(item.amodalLogits, 1, leafThr, item.w, item.h);
    item.leafMetrics = leafMetricsFromMask(leafMask, item.w, item.h, 0);
    formatLeafMetrics(item);
  } else {
    formatLeafMetrics(item);
  }
  updateAmodalQuality(item);
  const counts = paintAmodalWithMetrics(item);
  showCanvas(ui.cAmodal, item.amodalCanvas);
  if (counts) {
    const thrTxt = settings.map((s) => (s.enabled ? s.thr.toFixed(2) : "off")).join("/");
    ui.amodalMeta.textContent = t("metaLayers");
  }
}

function scheduleScoreRefresh() {
  const item = current();
  if (!item?.done || !state.stone || !state.amodal || state.batchRunning || state.refreshingScore) return;
  clearTimeout(state.scoreRefreshTimer);
  state.scoreRefreshTimer = setTimeout(() => { void refreshScoreByRerun(); }, 450);
}

async function refreshScoreByRerun() {
  const item = current();
  if (!item?.done || !state.stone || !state.amodal || state.batchRunning || state.refreshingScore) return;
  const confThr = Number(ui.scoreThr.value);
  if (item.scoreUsed != null && Math.abs(item.scoreUsed - confThr) < 1e-6) return;
  state.refreshingScore = true;
  updateReady();
  try {
    log(`conf→${confThr.toFixed(2)} · 重跑 YOLOE+Amodal…`);
    await runOnItem(item);
    await showIndex(state.index);
  } catch (e) {
    log(`重跑失败: ${e.message || e}`, "err");
  } finally {
    state.refreshingScore = false;
    updateReady();
  }
}

function renderThumbs() {
  ui.thumbStrip.innerHTML = "";
  state.album.forEach((item, i) => {
    const img = document.createElement("img");
    img.className = "thumb" + (i === state.index ? " active" : "") + (item.done ? " done" : "");
    img.src = item.thumbUrl || item.url || "";
    img.alt = item.name;
    img.title = item.name;
    img.loading = "lazy";
    img.onerror = () => {
      img.classList.add("broken");
      img.removeAttribute("src");
      img.alt = "!";
      img.title = t("thumbBroken", { name: item.name });
    };
    img.addEventListener("click", () => { void showIndex(i); });
    ui.thumbStrip.appendChild(img);
  });
  const nDone = state.album.filter((x) => x.done).length;
  ui.albumHint.textContent = state.album.length
    ? t("albumCount", { n: state.album.length }) + (nDone ? t("readyDone", { done: nDone, total: state.album.length }) : "")
    : t("albumEmpty");
  ui.navInfo.innerHTML = state.album.length
    ? `<strong>${state.index + 1}</strong> / ${state.album.length}　${current()?.name || ""}`
    : "0 / 0";
}

async function ensureBitmap(item) {
  if (item.bitmap) return item.bitmap;
  let raw;
  if (item.file) raw = await createImageBitmap(item.file);
  else if (item.url) {
    const res = await fetch(item.url);
    if (!res.ok) throw new Error(`HTTP ${res.status} · ${item.name}`);
    raw = await createImageBitmap(await res.blob());
  } else throw new Error(`无图片源: ${item.name}`);
  item.bitmap = await limitBitmapSide(raw, MAX_SIDE);
  item.w = item.bitmap.width;
  item.h = item.bitmap.height;
  return item.bitmap;
}

async function showIndex(i) {
  if (i < 0 || i >= state.album.length) return;
  state.index = i;
  const item = state.album[i];
  const bitmap = await ensureBitmap(item);
  ensureRoi(item);
  showCanvas(ui.cOrig, bitmap);
  paintRoiOverlay(item);
  ui.imgMeta.textContent = `${item.w}×${item.h}`;
  if (item.stoneCanvas) {
    showCanvas(ui.cStone, item.stoneCanvas);
    ui.stoneMeta.textContent = item.topScore != null
      ? t("metaStoneScore", { score: item.topScore.toFixed(3) })
      : "";
  } else {
    showCanvas(ui.cStone, null);
    ui.stoneMeta.textContent = t("metaUntreated");
  }
  if (item.amodalCanvas) {
    showCanvas(ui.cAmodal, item.amodalCanvas);
    ui.amodalMeta.textContent = item.stoneEmpty ? t("metaNoStone") : t("metaLayers");
  } else {
    showCanvas(ui.cAmodal, null);
    ui.amodalMeta.textContent = t("metaUntreated");
  }
  formatLeafMetrics(item);
  formatAmodalQuality(item);
  renderThumbs();
  updateReady();
}

async function loadSamples() {
  try {
    const names = await (await fetch(`${SAMPLES_DIR}/manifest.json`)).json();
    for (const name of names) {
      const url = `${SAMPLES_DIR}/${encodeURIComponent(name)}`;
      state.album.push({ name, url, thumbUrl: url, done: false });
    }
    log(t("logSamples", { n: names.length }), "ok");
    await showIndex(0);
  } catch (e) {
    log(t("logSamplesFail", { e: e.message || e }), "err");
  }
}

async function addFiles(fileList, { fromCamera = false } = {}) {
  const files = [...fileList].filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  for (const file of files) {
    let name = file.name || "capture.jpg";
    if (fromCamera && (!file.name || /^image\./i.test(file.name))) {
      const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      name = `camera_${ts}.jpg`;
    }
    state.album.push({
      name,
      file,
      thumbUrl: URL.createObjectURL(file),
      done: false,
    });
  }
  log(
    `fromCamera
      ? t("logCamera", { n: files.length, total: state.album.length })
      : t("logAdd", { n: files.length, total: state.album.length })`,
    "ok",
  );
  // After camera capture (or first add), select the newest image.
  if (fromCamera || state.index < 0) {
    await showIndex(state.album.length - 1);
  } else {
    renderThumbs();
    updateReady();
  }
}

async function urlExists(url) {
  try {
    const r = await fetch(url, { method: "HEAD" });
    if (r.ok) return true;
    const g = await fetch(url, { method: "GET", headers: { Range: "bytes=0-0" } });
    return g.ok || g.status === 206;
  } catch {
    return false;
  }
}

async function loadModels() {
  ui.btnLoad.disabled = true;
  const prevLabel = ui.btnLoad.textContent;
  ui.btnLoad.textContent = t("btnLoading");
  try {
    if (!state.ort) await importOrt();
    if (!state.stone) {
      let loaded = false;
      for (const u of STONE_URLS) {
        if (!(await urlExists(u))) continue;
        log(t("logLoadStone"));
        try {
          const preferStone = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
          const stone = await createSession(u, u, preferStone);
          state.stone = stone.session;
          state.epStone = stone.ep;
          state.stoneUrl = u;
          state.yoloeMeta = null;
          if (/yoloe_stone_ruler/i.test(u)) {
            try {
              const mr = await fetch(YOLOE_META_URL);
              if (mr.ok) {
                state.yoloeMeta = await mr.json();
                log(
                  `  多类: stone=${JSON.stringify(state.yoloeMeta.stone_ids)} ` +
                    `ruler=${JSON.stringify(state.yoloeMeta.ruler_ids)}`,
                  "ok",
                );
              }
            } catch (me) {
              log(`  读取 ${YOLOE_META_URL} 失败: ${me.message || me}`, "err");
            }
          }
          loaded = true;
          break;
        } catch (e) {
          log(t("logSkipUrl", { u, e: e.message || e }), "err");
          state.stone = null;
        }
      }
      if (!loaded) throw new Error(t("errNoStoneModel"));
    }
    if (!state.amodal) {
      const prefer = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
      log(t("logLoadLeaf"));
      state.amodalBytes = await loadModelBytes(AMODAL_MANIFEST, AMODAL_URL, "Amodal Q4");
      const amodal = await createSession(state.amodalBytes, "amodal_dino_q4", prefer);
      state.amodal = amodal.session;
      state.epAmodal = amodal.ep;
    }
    log("预热石头模型…");
    {
      const zeros = new Float32Array(1 * 3 * YOLOE_SIZE * YOLOE_SIZE);
      const t0 = performance.now();
      await state.stone.run({
        images: new state.ort.Tensor("float32", zeros, [1, 3, YOLOE_SIZE, YOLOE_SIZE]),
      });
      log(`  YOLOE 预热 ${(performance.now() - t0).toFixed(0)} ms`, "ok");
    }
    if (state.epAmodal === "webgpu") {
      log("YOLOE 预热后重建 Amodal session…");
      await recreateAmodalSession(["webgpu", "wasm"]);
    }
    const tag = state.yoloeMeta ? "YOLOE+ruler+Q4" : "YOLOE+Q4";
    ui.badge.textContent =
      `${tag} · S:${(state.epStone || "?").toUpperCase()} · A:${(state.epAmodal || "?").toUpperCase()}`;
    ui.badge.classList.add("ok");
    log(t("logModelsReady"), "ok");
    ui.btnLoad.textContent = t("btnReload");
  } catch (e) {
    log(t("logModelsFail", { e: e.message || e }), "err");
    ui.badge.textContent = "ERROR";
    ui.badge.classList.remove("ok");
    ui.btnLoad.textContent = prevLabel || t("btnLoad");
  } finally {
    ui.btnLoad.disabled = false;
    updateReady();
  }
}

async function runOnItem(item, { silent = false } = {}) {
  const ort = state.ort;
  const confThr = Number(ui.scoreThr.value);
  const bitmap = await ensureBitmap(item);
  const roi = ensureRoi(item);

  if (!silent) log(t("logProcess", { name: item.name }));
  const meta = letterboxYoloe(bitmap);
  let t0 = performance.now();
  const stoneOut = await state.stone.run({
    images: new ort.Tensor("float32", meta.chw, [1, 3, YOLOE_SIZE, YOLOE_SIZE]),
  });
  const stoneMs = performance.now() - t0;
  const outNames = state.stone.outputNames;
  const det = tensorToFloat32((stoneOut.output0 || stoneOut[outNames[0]]).data);
  const proto = tensorToFloat32((stoneOut.output1 || stoneOut[outNames[1]]).data);
  const stoneIds = state.yoloeMeta?.stone_ids
    ? new Set(state.yoloeMeta.stone_ids)
    : null;
  const { soft, topScore } = decodeYoloeMask(det, proto, meta, confThr, stoneIds);
  const empty = topScore <= 0 || isStoneMaskEmpty(soft, bitmap.width, bitmap.height);
  item.stoneSoft = soft;
  item.topScore = topScore;
  item.scoreUsed = confThr;
  item.stoneEmpty = empty;
  item.rulerInstances = null;
  item.scale = null;
  item.leafMetrics = null;
  if (state.yoloeMeta?.ruler_ids) {
    const rulerConf = Math.min(confThr, Number(state.yoloeMeta.default_conf || 0.08));
    item.rulerInstances = decodeYoloeClassInstances(
      det,
      proto,
      meta,
      rulerConf,
      state.yoloeMeta.ruler_ids,
    );
    if (!silent && item.rulerInstances.length) {
      log(`  rulers ${item.rulerInstances.length}`, "ok");
    }
  }
  if (empty) {
    item.stoneCanvas = makeSolidWhiteCanvas(bitmap.width, bitmap.height);
    applyNoStoneSkipAmodal(item);
    item.done = true;
    if (!silent) {
      log(`  YOLOE ${stoneMs.toFixed(0)} ms · conf=${topScore.toFixed(3)}`, "ok");
      log(t("logNoStone"), "err");
    }
    return { stoneMs, amodalMs: 0, topScore, empty: true };
  }

  const maskCanvas = binarySoftToCanvas(soft, bitmap.width, bitmap.height);
  const stoneCanvas = makeWhiteBgStone(bitmap, maskCanvas);
  item.stoneCanvas = stoneCanvas;
  if (!silent) log(`  YOLOE ${stoneMs.toFixed(0)} ms · conf=${topScore.toFixed(3)}`, "ok");

  const input = buildAmodalInput(stoneCanvas, roi, item.w, item.h);
  const feed = { rgb_roi: new ort.Tensor("float32", input, [1, 4, AMODAL_SIZE, AMODAL_SIZE]) };

  async function inferAmodal(epLabel) {
    const t1 = performance.now();
    const amodalOut = await state.amodal.run(feed);
    const amodalMs = performance.now() - t1;
    const logits = tensorToFloat32((amodalOut.logits || amodalOut[state.amodal.outputNames[0]]).data);
    const check = amodalLogitsLookBad(logits);
    if (!silent) {
      log(
        `  Amodal ${amodalMs.toFixed(0)} ms · ep=${epLabel} · span=${check.span.toFixed(1)} · vis=${(check.visFrac * 100).toFixed(1)}%` +
          (check.bad ? ` · BAD(${check.reason})` : ""),
        check.bad ? "err" : "ok",
      );
    }
    return { amodalMs, logits, check };
  }

  let out = await inferAmodal(state.epAmodal || "?");
  if (out.check.bad && state.epAmodal === "webgpu") {
    log("  Amodal WebGPU 数值异常 → 重建并重试…", "err");
    await recreateAmodalSession(["webgpu", "wasm"]);
    out = await inferAmodal(state.epAmodal || "?");
  }
  if (out.check.bad && state.epAmodal === "webgpu") {
    log("  仍异常 → 回退 WASM（会明显变慢）…", "err");
    await recreateAmodalSession(["wasm"]);
    out = await inferAmodal(state.epAmodal || "?");
  }

  item.amodalLogits = out.logits;
  try {
    updateLeafMetrics(item);
  } catch (e) {
    if (!silent) log(t("logScaleFail", { e: e.message || e }), "err");
  }
  updateAmodalQuality(item);
  if (!silent) {
    const q = item.amodalQuality?.amodal;
    if (q && Number.isFinite(q.quality_score)) {
      log(
        `  quality leaf Q=${q.quality_score.toFixed(3)} · in=${q.mean_prob_in.toFixed(2)} out=${q.mean_prob_out.toFixed(2)} · sym=${Number.isFinite(q.symmetry) ? q.symmetry.toFixed(2) : "n/a"}`,
        "ok",
      );
    }
  }
  paintAmodalWithMetrics(item);
  item.done = true;
  if (out.check.bad && !silent) log("  警告: amodal 输出仍异常", "err");
  return { stoneMs, amodalMs: out.amodalMs, topScore, empty: false };
}

async function runCurrent() {
  const item = current();
  if (!item || !state.stone || !state.amodal) return;
  ui.btnRun.disabled = true;
  try {
    await runOnItem(item);
    await showIndex(state.index);
  } catch (e) {
    log(`运行失败: ${e.message || e}`, "err");
  } finally {
    updateReady();
  }
}

async function runBatch() {
  if (!state.stone || !state.amodal || !state.album.length || state.batchRunning) return;
  state.batchRunning = true;
  updateReady();
  const tAll = performance.now();
  let ok = 0;
  log(t("logBatchStart", { n: state.album.length }));
  try {
    for (let i = 0; i < state.album.length; i++) {
      log(t("logBatchItem", { i: i + 1, n: state.album.length, name: state.album[i].name }));
      await showIndex(i);
      try {
        await runOnItem(state.album[i], { silent: true });
        ok += 1;
        log(`  ✓ conf=${state.album[i].topScore.toFixed(3)}`, "ok");
        await showIndex(i);
      } catch (e) {
        log(`  ✗ ${e.message || e}`, "err");
      }
    }
    log(t("logBatchDone", { ok, total: state.album.length, s: ((performance.now() - tAll) / 1000).toFixed(1) }), "ok");
  } finally {
    state.batchRunning = false;
    updateReady();
  }
}

ui.btnLoad.addEventListener("click", () => {
  state.stone = null;
  state.amodal = null;
  state.amodalBytes = null;
  state.epAmodal = null;
  state.epStone = null;
  state.stoneUrl = null;
  state.yoloeMeta = null;
  void loadModels();
});
ui.btnRun.addEventListener("click", runCurrent);
ui.btnBatch.addEventListener("click", runBatch);
ui.btnPrev.addEventListener("click", () => showIndex(state.index - 1));
ui.btnNext.addEventListener("click", () => showIndex(state.index + 1));
ui.btnDraw.addEventListener("click", () => {
  state.drawMode = state.drawMode === "draw" ? null : "draw";
  updateReady();
});
ui.btnErase.addEventListener("click", () => {
  state.drawMode = state.drawMode === "erase" ? null : "erase";
  updateReady();
});
ui.btnRoiReset.addEventListener("click", () => {
  const item = current();
  if (!item) return;
  ensureRoi(item).fill(ROI_NEUTRAL);
  item.roiEdited = false;
  paintRoiOverlay(item);
  log(t("logRoiReset"));
});
ui.btnRoiClear.addEventListener("click", () => {
  const item = current();
  if (!item) return;
  ensureRoi(item).fill(ROI_NEG);
  item.roiEdited = true;
  paintRoiOverlay(item);
  log(t("logRoiClear"));
});
ui.imageFile.addEventListener("change", () => {
  if (ui.imageFile.files?.length) {
    addFiles(ui.imageFile.files);
    ui.imageFile.value = "";
  }
});
ui.cameraFile?.addEventListener("change", () => {
  if (ui.cameraFile.files?.length) {
    addFiles(ui.cameraFile.files, { fromCamera: true });
    ui.cameraFile.value = "";
  }
});
window.addEventListener("keydown", (ev) => {
  if (ev.target && ["INPUT", "TEXTAREA"].includes(ev.target.tagName)) return;
  if (ev.key === "ArrowLeft") showIndex(state.index - 1);
  if (ev.key === "ArrowRight") showIndex(state.index + 1);
});

bindRoiDrawing();
initLang();
document.getElementById("btnLangZh")?.addEventListener("click", () => {
  setLang("zh");
  refreshLangUi();
});
document.getElementById("btnLangEn")?.addEventListener("click", () => {
  setLang("en");
  refreshLangUi();
});
log(t("logOrtReady", { gpu: !!navigator.gpu }));
loadSamples();

/** Local pack with full .onnx → auto-load; GitHub Pages (chunks only) → wait for click. */
(async () => {
  const hasFullAmodal = await urlExists(AMODAL_URL);
  if (hasFullAmodal) {
    ui.btnLoad.textContent = t("btnReload");
    log(t("logAutoLoad"));
    await loadModels();
  } else {
    ui.btnLoad.textContent = t("btnLoad");
    ui.btnLoad.disabled = false;
    log(t("logWaitClick"), "ok");
  }
})();
