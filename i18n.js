/** UI strings: plain language (zh / en). */

export const LANGS = ["zh", "en"];

const STR = {
  zh: {
    pageTitle: "树叶化石 · 在线分析",
    headerTitle: "树叶化石补全",
    headerSub: "① 找出石头 → ② 画出叶子轮廓和叶脉 · 可在浏览器里离线运行",
    sectionModel: "模型",
    btnLoad: "加载分析模型",
    btnLoading: "加载中…",
    btnReload: "重新加载模型",
    hintLoad: "本机已有完整模型文件时会自动加载；网上打开时请点按钮（大约要下载 300MB）",
    labelUploadStone: "或手动选择「找石头」模型文件",
    labelUploadAmodal: "或手动选择「画叶子」模型文件",
    btnChooseFile: "选择文件",
    sectionImages: "图片",
    labelUploadImages: "从相册选择图片",
    labelCamera: "现场拍照",
    btnTakePhoto: "打开相机",
    hintCamera: "手机上会打开后置摄像头；拍完后自动显示这张照片",
    albumEmpty: "还没有图片",
    albumCount: "共 {n} 张",
    btnRun: "分析当前这张",
    btnBatch: "批量分析全部",
    readyHint: "准备情况：找石头 {s}　画叶子 {a}　图片 {i}",
    readyEdit: "　｜　改参数会自动刷新",
    btnDlStone: "下载白底石头图",
    btnDlOverlay: "下载彩色叠加图",
    legendVis: "可见部分",
    legendLeaf: "完整叶片",
    legendVein: "主叶脉",
    legendDetail: "细叶脉",
    statusBoot: "正在启动…",
    hintServer: "电脑上请双击 start_server.bat 打开本页，不要直接双击 html 文件",
    capOrig: "① 原图 · 可涂抹ROI区域",
    capStone: "② 抠出的石头（白底）",
    capAmodal: "③ 叶子与叶脉叠加",
    roiTitle: "涂抹ROI区域",
    btnDraw: "画笔",
    btnErase: "橡皮",
    brushSize: "笔刷大小",
    btnRoiReset: "恢复为默认（中性 0.5）",
    btnRoiClear: "全部标为忽略（0）",
    hintRoi: "左键涂青=关注(1)，右键涂红=忽略(0)。未涂抹处为中性 0.5，无需先清空。",
    stoneThrTitle: "找石头的灵敏度",
    scoreLabel: "置信度（越高越严格）",
    maskLabel: "轮廓软硬（备用）",
    hintStoneThr: "拖动后会自动重算。一般只调「置信度」即可。",
    amodalThrTitle: "各层显示开关与灵敏度",
    headVis: "可见部分",
    headLeaf: "完整叶片",
    headVein: "主叶脉",
    headDetail: "细叶脉",
    hintAmodalThr: "勾选是否显示 · 点色块改颜色 · 拖动滑条立刻更新，不用重新跑模型",
    sizeTitle: "叶片尺寸（靠图里的尺子）",
    enLeafBox: "在图上画出叶片外框和尺寸",
    metricsNone: "尺寸：未计算",
    qualityNone: "质量：未评估",
    hintQuality: "质量分：看轮廓是否清晰、形状是否规整（改灵敏度会重算）。",
    hintScale: "若照片里有厘米尺，会自动估算出叶片长、宽、面积。",
    metaUntreated: "尚未分析",
    metaNoStone: "未找到石头 · 已跳过画叶",
    metaLayers: "可见 / 完整叶 / 叶脉",
    thumbBroken: "{name} 打不开（若在网上：请确认已上传 .nojekyll）",
    langZh: "中文",
    langEn: "EN",
    // dynamic / logs (plain)
    logBoot: "页面已打开 · 显卡加速={gpu}",
    logAutoLoad: "发现完整模型文件，开始自动加载…",
    logWaitClick: "页面已就绪。请点击「加载分析模型」（首次下载可能较久）。",
    logSamples: "示例相册 {n} 张",
    logSamplesFail: "示例图加载失败: {e}",
    logAdd: "已加入 {n} 张，一共 {total} 张",
    logCamera: "拍照加入 {n} 张，一共 {total} 张",
    logModelsOk: "模型已就绪，可以开始分析",
    logModelsFail: "模型加载失败: {e}",
    logProcess: "正在分析: {name}",
    logNoStone: "这张图没找到石头，已跳过画叶",
    logDone: "分析完成",
    logBatchDone: "批量完成 {ok}/{total} · 用时 {s}s",
    sizeSkipped: "尺寸：未找到石头，已跳过",
    qualitySkipped: "质量：已跳过",
    sizePending: "尺寸：未计算",
    readyDone: " · 已完成 {done}/{total}",
    metaStoneScore: "把握度 {score}",
    metaStoneScoreThr: "把握度 {score} · 设定 {thr}",
    sizeNoRuler: "图里没找到尺子",
    sizeScale: "约 {ppc} 像素/厘米",
    sizeCm: "长 {l} cm · 宽 {w} cm · 面积 {a} cm²",
    sizePx: "长 {l} px · 宽 {w} px · 面积 {a} px²",
    sizeLabel: "尺寸：{body}",
    qualityNoneDyn: "质量：未评估",
    qualityEmpty: "质量：几乎没画出来",
    qualityLeaf: "叶片",
    qualityVein: "叶脉",
    qualityFmt: "质量：{parts}",
    qualityPart: "{name} {score}",
    logRoiReset: "ROI区域已恢复为中性（0.5）",
    logRoiClear: "整图已标为忽略（0），可用画笔涂出关注区",
    logScaleFail: "  尺寸估算失败: {e}",
    logScaleOk: "  已根据尺子估算尺寸",
    logScaleMiss: "  没找到可用的尺子刻度",
    logWarmOk: "  预热完成，用时 {ms} ms",
    logWarmFail: "  预热失败: {e}",
    logSkipUrl: "  跳过 {u}: {e}",
    logLoadStone: "正在加载「找石头」模型…",
    logLoadLeaf: "正在加载「画叶子」模型…",
    logModelsReady: "两个模型都已就绪，可以开始分析",
    errNoStoneModel: "找不到可用的「找石头」模型文件",
    errNoLeafModel: "找不到可用的「画叶子」模型文件",
    errOrtHttp: "模型库加载失败（HTTP {status}）。请用 start_server.bat 打开本页",
    logResize: "  图片已缩小：{before} → {after}",
    logScoreRerun: "正在按新设定重新找石头并画叶…",
    logScoreFail: "按新设定重算失败: {e}",
    logBatchStart: "开始批量分析，共 {n} 张",
    logBatchItem: "批量 [{i}/{n}] {name}",
    logFail: "失败: {e}",
    logOrtReady: "运行环境已就绪 · 显卡加速={gpu}",
  },
  en: {
    pageTitle: "Leaf Fossil · Analyze",
    headerTitle: "Leaf Fossil Analyzer",
    headerSub: "① Find the stone → ② Draw the leaf & veins · Works offline in the browser",
    sectionModel: "Models",
    btnLoad: "Load models",
    btnLoading: "Loading…",
    btnReload: "Reload models",
    hintLoad: "Loads automatically if the full model file is on this device; otherwise tap the button (~300MB download)",
    labelUploadStone: "Or pick a “find stone” model file",
    labelUploadAmodal: "Or pick a “draw leaf” model file",
    btnChooseFile: "Choose file",
    sectionImages: "Photos",
    labelUploadImages: "Choose from gallery",
    labelCamera: "Take a photo",
    btnTakePhoto: "Open camera",
    hintCamera: "Opens the rear camera on phones; the new photo is selected automatically",
    albumEmpty: "No photos yet",
    albumCount: "{n} photo(s)",
    btnRun: "Analyze this photo",
    btnBatch: "Analyze all",
    readyHint: "Ready: stone {s}　leaf {a}　photo {i}",
    readyEdit: "　｜　tweaking sliders refreshes the view",
    btnDlStone: "Download white-background stone",
    btnDlOverlay: "Download color overlay",
    legendVis: "Visible part",
    legendLeaf: "Full leaf",
    legendVein: "Main veins",
    legendDetail: "Fine veins",
    statusBoot: "Starting…",
    hintServer: "On a PC, double-click start_server.bat — don’t open the html file directly",
    capOrig: "① Original · paint region of interest",
    capStone: "② Cut-out stone (white bg)",
    capAmodal: "③ Leaf & vein overlay",
    roiTitle: "Region of interest",
    btnDraw: "Brush",
    btnErase: "Eraser",
    brushSize: "Brush size",
    btnRoiReset: "Reset to neutral (0.5)",
    btnRoiClear: "Mark all ignored (0)",
    hintRoi: "Left-click cyan = focus (1), right-click red = ignore (0). Untouched areas stay neutral 0.5.",
    stoneThrTitle: "Stone detection sensitivity",
    scoreLabel: "Confidence (higher = stricter)",
    maskLabel: "Edge softness (fallback)",
    hintStoneThr: "Updates automatically. Usually only “Confidence” matters.",
    amodalThrTitle: "Layer visibility & sensitivity",
    headVis: "Visible part",
    headLeaf: "Full leaf",
    headVein: "Main veins",
    headDetail: "Fine veins",
    hintAmodalThr: "Toggle layers · pick colors · drag sliders to refresh (no need to re-run)",
    sizeTitle: "Leaf size (from ruler in photo)",
    enLeafBox: "Show leaf box & size on the overlay",
    metricsNone: "Size: not estimated yet",
    qualityNone: "Quality: not scored yet",
    hintQuality: "Quality score: how clear and tidy the outline looks (recomputed when you move sliders).",
    hintScale: "If a cm ruler is in the photo, length / width / area are estimated automatically.",
    metaUntreated: "Not analyzed yet",
    metaNoStone: "No stone found · leaf step skipped",
    metaLayers: "visible / full leaf / veins",
    thumbBroken: "{name} failed to load (online: make sure .nojekyll is uploaded)",
    langZh: "中文",
    langEn: "EN",
    logBoot: "Page ready · GPU accel={gpu}",
    logAutoLoad: "Full model file found — loading automatically…",
    logWaitClick: "Page ready. Tap “Load models” (first download may take a while).",
    logSamples: "Sample album: {n} photo(s)",
    logSamplesFail: "Failed to load samples: {e}",
    logAdd: "Added {n}; album has {total}",
    logCamera: "Camera added {n}; album has {total}",
    logModelsOk: "Models ready — you can analyze photos",
    logModelsFail: "Model load failed: {e}",
    logProcess: "Analyzing: {name}",
    logNoStone: "No stone in this photo — leaf step skipped",
    logDone: "Done",
    logBatchDone: "Batch done {ok}/{total} · {s}s",
    sizeSkipped: "Size: skipped (no stone)",
    qualitySkipped: "Quality: skipped",
    sizePending: "Size: not estimated yet",
    readyDone: " · done {done}/{total}",
    metaStoneScore: "confidence {score}",
    metaStoneScoreThr: "confidence {score} · set {thr}",
    sizeNoRuler: "no ruler found in photo",
    sizeScale: "about {ppc} px per cm",
    sizeCm: "L {l} cm · W {w} cm · area {a} cm²",
    sizePx: "L {l} px · W {w} px · area {a} px²",
    sizeLabel: "Size: {body}",
    qualityNoneDyn: "Quality: not scored yet",
    qualityEmpty: "Quality: almost empty",
    qualityLeaf: "leaf",
    qualityVein: "veins",
    qualityFmt: "Quality: {parts}",
    qualityPart: "{name} {score}",
    logRoiReset: "ROI reset to neutral (0.5)",
    logRoiClear: "Whole image marked ignore (0) — paint focus areas with the brush",
    logScaleFail: "  Size estimate failed: {e}",
    logScaleOk: "  Size estimated from ruler",
    logScaleMiss: "  No usable ruler ticks found",
    logWarmOk: "  Warm-up done in {ms} ms",
    logWarmFail: "  Warm-up failed: {e}",
    logSkipUrl: "  Skip {u}: {e}",
    logLoadStone: "Loading “find stone” model…",
    logLoadLeaf: "Loading “draw leaf” model…",
    logModelsReady: "Both models ready — you can analyze photos",
    errNoStoneModel: "No usable “find stone” model file found",
    errNoLeafModel: "No usable “draw leaf” model file found",
    errOrtHttp: "Runtime load failed (HTTP {status}). Open via start_server.bat",
    logResize: "  Image resized: {before} → {after}",
    logScoreRerun: "Re-finding stone and redrawing leaf…",
    logScoreFail: "Re-run failed: {e}",
    logBatchStart: "Batch start · {n} photo(s)",
    logBatchItem: "Batch [{i}/{n}] {name}",
    logFail: "Failed: {e}",
    logOrtReady: "Runtime ready · GPU accel={gpu}",
  },
};

/** @type {'zh'|'en'} */
let lang = "zh";

export function getLang() {
  return lang;
}

export function t(key, vars) {
  const table = STR[lang] || STR.zh;
  let s = table[key] ?? STR.zh[key] ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      s = s.replaceAll(`{${k}}`, String(v));
    }
  }
  return s;
}

export function setLang(next) {
  lang = next === "en" ? "en" : "zh";
  try {
    localStorage.setItem("leaf_fossil_lang", lang);
  } catch { /* ignore */ }
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  applyStaticI18n();
  return lang;
}

export function initLang() {
  let saved = null;
  try {
    saved = localStorage.getItem("leaf_fossil_lang");
  } catch { /* ignore */ }
  if (saved === "en" || saved === "zh") lang = saved;
  else if (typeof navigator !== "undefined" && navigator.language && !navigator.language.toLowerCase().startsWith("zh")) {
    lang = "en";
  }
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  applyStaticI18n();
  return lang;
}

/** Apply data-i18n / data-i18n-html on elements. */
export function applyStaticI18n(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (!key) return;
    const text = t(key);
    if (el.tagName === "TITLE") {
      document.title = text;
      return;
    }
    // Preserve nested <input> inside labels: only update text nodes / leading span
    const span = el.querySelector(":scope > span.i18n-label, :scope > span:not(.badge)");
    if (span && el.matches("label.field, label.name")) {
      span.textContent = text;
      return;
    }
    if (el.childElementCount && el.querySelector("input, code, i, strong, span.badge")) {
      // Replace only direct text, or first text-bearing child marked
      const labeled = el.querySelector("[data-i18n-self]");
      if (labeled) {
        labeled.textContent = text;
        return;
      }
    }
    el.textContent = text;
  });
  root.querySelectorAll("[data-i18n-html]").forEach((el) => {
    el.innerHTML = t(el.getAttribute("data-i18n-html"));
  });
  root.querySelectorAll("[data-i18n-title]").forEach((el) => {
    el.title = t(el.getAttribute("data-i18n-title"));
  });
  const btnZh = document.getElementById("btnLangZh");
  const btnEn = document.getElementById("btnLangEn");
  if (btnZh && btnEn) {
    btnZh.classList.toggle("active", lang === "zh");
    btnEn.classList.toggle("active", lang === "en");
  }
}
