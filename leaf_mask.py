import os
import random
import shutil
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import concurrent.futures  # 引入并发库

# ==================== 全局配置常量 ====================
SAVE_ELLIPSE_MASK = False  # 是否保存原始的纯椭圆遮罩图 (True/False)
COPY_UNDAMAGED_MASKS = False  # 是否为 amodal/vein 等不挖洞模态创建 broken 副本
PROCESS_DIFFUSION = False  # 是否同步处理 leaf_fossil/diffusion 文件夹
STONE_BG_PROB = 0.333       # ellipse/irregular 用 stone 背景填充挖洞区的概率
RATIO_TYPE = "20-60"       # 遮罩大小比例，可选 '20-40'、'40-60' 或 '20-60'
MIN_RETAIN_RATIO = 0.333    # 最少保留的叶片面积比例 (1/3)
NUM_WORKERS = 16           # 线程池数量，设置为 None 时将自动匹配 CPU 核心数

# ----------------- 阴影与物体提取控制 -----------------
OBJECT_THRESHOLD = 250     # 提取有效物体的灰度阈值（低于此值视为非纯白物体）
SHADOW_SIZE = 5           # 阴影向外扩散的大小（像素宽度，常量控制）
SHADOW_DARKNESS = 0.5     # 阴影最深处的暗度（0.0为全黑，1.0为无阴影）

# ----------------- 重复生成与重试控制 -----------------
REPEAT_COUNT = 4           # 每张图反复运行生成的数量。默认为 1
MAX_ATTEMPTS = 8           # 【新增】单张图不达标时，最多重新尝试生成遮罩的次数
AREA_CHECK_THRESHOLD = 0.75 # 【新增】判定是否成功挖洞的阈值（保留面积必须小于 75%）

# ----------------- 不规则遮罩形状控制 -----------------
# 以下强度均为相对遮罩等效半径的比例，每次生成在区间内随机采样
IRREGULAR_WARP_STRENGTH = (0.06, 0.22)      # 边缘整体扭曲幅度
IRREGULAR_BUMP_STRENGTH = (0.04, 0.16)      # 轮廓凸起/凹陷幅度
IRREGULAR_BUMP_COUNT = (1, 3)               # 凸起数量（越少越稀疏）
IRREGULAR_BUMP_WIDTH_FRAC = (0.12, 0.28)    # 凸起宽度（相对轮廓周长比例，越大越稀疏）
IRREGULAR_BUMP_MIN_SPACING_FRAC = 0.14      # 凸起中心最小间距（相对轮廓周长比例）
IRREGULAR_NOISE_STRENGTH = (0.02, 0.10)     # 高频边缘抖动
IRREGULAR_MORPH_STRENGTH = (0.0, 0.08)      # 形态学侵蚀/膨胀扰动
CONTOUR_SAMPLE_POINTS = 160                 # 轮廓重采样点数，越大边缘越细腻

# ----------------- 直线遮罩控制 -----------------
LINE_THICKNESS = (60, 120)                  # 直线粗细（256 基准像素，随分辨率缩放）
LINE_DISTORT_PASSES = 2                     # 直线不规则扭曲迭代次数
# 直线专用不规则强度（通常高于椭圆/不规则遮罩）
LINE_IRREGULAR_WARP_STRENGTH = (0.14, 0.38)
LINE_IRREGULAR_BUMP_STRENGTH = (0.10, 0.28)
LINE_IRREGULAR_BUMP_COUNT = (2, 4)
LINE_IRREGULAR_BUMP_WIDTH_FRAC = (0.10, 0.24)
LINE_IRREGULAR_BUMP_MIN_SPACING_FRAC = 0.12
LINE_IRREGULAR_NOISE_STRENGTH = (0.06, 0.20)
LINE_IRREGULAR_MORPH_STRENGTH = (0.05, 0.16)

# 三种遮罩方法轮流使用：ellipse | irregular | line
MASK_METHODS = ("ellipse", "irregular", "line")
MASK_METHOD_NAMES = {
    "ellipse": "椭圆",
    "irregular": "不规则",
    "line": "直线",
}

# ----------------- leaf_fossil 同步复制 -----------------
FOSSIL_LEAF_DIR = r"E:\Desktop\ji\490model\leaf_fossil\diffusion\leaf"
FOSSIL_OUT_DIR = r"E:\Desktop\ji\490model\leaf_fossil\diffusion\leaf_fossil"
# ======================================================

def _image_scale(img_shape):
    """基于 256x256 基准分辨率计算缩放因子"""
    H, W = img_shape[:2]
    return ((W / 256.0) + (H / 256.0)) / 2.0


def _smooth_1d_signal(values, window):
    window = max(3, int(window) | 1)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _resample_contour(pts, n_samples):
    """按弧长均匀重采样闭合轮廓"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return np.tile(pts[0], (n_samples, 1)) if len(pts) else np.zeros((n_samples, 2))

    closed = np.vstack([pts, pts[0]])
    deltas = np.diff(closed, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n_samples, 1))

    targets = np.linspace(0.0, total, n_samples, endpoint=False)
    resampled = np.zeros((n_samples, 2), dtype=np.float64)
    seg_idx = 0
    for i, target in enumerate(targets):
        while seg_idx < len(seg_lens) - 1 and cumlen[seg_idx + 1] < target:
            seg_idx += 1
        seg_len = seg_lens[seg_idx]
        alpha = 0.0 if seg_len < 1e-6 else (target - cumlen[seg_idx]) / seg_len
        resampled[i] = closed[seg_idx] * (1.0 - alpha) + closed[seg_idx + 1] * alpha
    return resampled


def _sample_strength(strength_range):
    """在 (min, max) 区间内随机采样强度"""
    return random.uniform(strength_range[0], strength_range[1])


def _place_bump_centers(count, n_points, min_spacing):
    """在轮廓上放置彼此保持最小间距的凸起中心索引"""
    centers = []
    max_attempts = max(30, count * 25)
    for _ in range(max_attempts):
        if len(centers) >= count:
            break
        candidate = random.randint(0, n_points - 1)
        if all(
            min(abs(candidate - center), n_points - abs(candidate - center)) >= min_spacing
            for center in centers
        ):
            centers.append(candidate)
    return centers


def distort_mask_irregular(
    base_mask,
    scale,
    warp_range=None,
    bump_range=None,
    bump_count_range=None,
    bump_width_frac_range=None,
    bump_min_spacing_frac=None,
    noise_range=None,
    morph_range=None,
):
    """
    将基础遮罩扭曲为不规则破损形状：
    低频径向扭曲 + 局部凸起 + 高频抖动 + 可选形态学扰动
    可通过参数覆盖默认强度区间（直线遮罩使用更强的区间）。
    """
    warp_range = warp_range or IRREGULAR_WARP_STRENGTH
    bump_range = bump_range or IRREGULAR_BUMP_STRENGTH
    bump_count_range = bump_count_range or IRREGULAR_BUMP_COUNT
    bump_width_frac_range = bump_width_frac_range or IRREGULAR_BUMP_WIDTH_FRAC
    bump_min_spacing_frac = (
        IRREGULAR_BUMP_MIN_SPACING_FRAC
        if bump_min_spacing_frac is None
        else bump_min_spacing_frac
    )
    noise_range = noise_range or IRREGULAR_NOISE_STRENGTH
    morph_range = morph_range or IRREGULAR_MORPH_STRENGTH
    H, W = base_mask.shape[:2]
    if np.count_nonzero(base_mask) == 0:
        return base_mask

    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return base_mask

    out = np.zeros((H, W), dtype=np.uint8)
    warp_strength = _sample_strength(warp_range)
    bump_strength = _sample_strength(bump_range)
    noise_strength = _sample_strength(noise_range)
    morph_strength = _sample_strength(morph_range)
    bump_count = random.randint(bump_count_range[0], bump_count_range[1])
    min_bump_spacing = max(8, int(CONTOUR_SAMPLE_POINTS * bump_min_spacing_frac))

    for contour in contours:
        if cv2.contourArea(contour) < 16:
            continue

        pts = contour.reshape(-1, 2).astype(np.float64)
        resampled = _resample_contour(pts, CONTOUR_SAMPLE_POINTS)
        centroid = resampled.mean(axis=0)
        vectors = resampled - centroid
        dists = np.linalg.norm(vectors, axis=1)
        equiv_radius = max(float(np.mean(dists)), 8.0 * scale)

        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        unit_vectors = vectors / (dists[:, None] + 1e-6)

        smooth_window = max(7, CONTOUR_SAMPLE_POINTS // 7)
        warp_noise = _smooth_1d_signal(np.random.randn(CONTOUR_SAMPLE_POINTS), smooth_window)
        warp_noise /= np.max(np.abs(warp_noise)) + 1e-6
        radial_offset = warp_noise * warp_strength * equiv_radius

        bump_offset = np.zeros(CONTOUR_SAMPLE_POINTS, dtype=np.float64)
        bump_centers = _place_bump_centers(bump_count, CONTOUR_SAMPLE_POINTS, min_bump_spacing)
        width_min = max(10, int(CONTOUR_SAMPLE_POINTS * bump_width_frac_range[0]))
        width_max = max(width_min + 4, int(CONTOUR_SAMPLE_POINTS * bump_width_frac_range[1]))
        for center_idx in bump_centers:
            width = random.randint(width_min, width_max)
            sign = 1.0 if random.random() < 0.75 else -0.6
            for i in range(CONTOUR_SAMPLE_POINTS):
                cyclic_dist = min(
                    abs(i - center_idx),
                    CONTOUR_SAMPLE_POINTS - abs(i - center_idx),
                )
                bump_offset[i] += sign * bump_strength * equiv_radius * np.exp(
                    -0.5 * (cyclic_dist / width) ** 2
                )

        hf_noise = _smooth_1d_signal(np.random.randn(CONTOUR_SAMPLE_POINTS), 5)
        hf_noise *= noise_strength * equiv_radius

        new_dists = np.clip(
            dists + radial_offset + bump_offset + hf_noise,
            equiv_radius * 0.35,
            equiv_radius * 2.2,
        )
        new_pts = centroid + unit_vectors * new_dists[:, None]
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, W - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, H - 1)
        cv2.fillPoly(out, [new_pts.astype(np.int32)], 255)

    if np.count_nonzero(out) == 0:
        return base_mask

    if morph_strength > 0.01:
        mask_radius = max(np.sqrt(np.count_nonzero(out) / np.pi), 8.0 * scale)
        ksize = max(3, int(morph_strength * mask_radius) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if random.random() < 0.5:
            out = cv2.dilate(out, kernel, iterations=1)
        else:
            out = cv2.erode(out, kernel, iterations=1)

    return out


def generate_line_damage_mask(img_shape):
    """生成贯穿全图的粗直线保留带，并施加加强版不规则边缘扭曲"""
    H, W = img_shape[:2]
    scale = _image_scale(img_shape)
    keep_band = np.zeros((H, W), dtype=np.uint8)

    angle = random.uniform(0, np.pi)
    cx, cy = W / 2.0, H / 2.0
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    length = float(max(H, W) * 2)
    p1 = (int(cx - direction[0] * length), int(cy - direction[1] * length))
    p2 = (int(cx + direction[0] * length), int(cy + direction[1] * length))

    thickness = int(random.randint(LINE_THICKNESS[0], LINE_THICKNESS[1]) * scale)
    thickness = max(5, thickness | 1)
    cv2.line(keep_band, p1, p2, 255, thickness, lineType=cv2.LINE_AA)

    line_distort_kwargs = {
        "warp_range": LINE_IRREGULAR_WARP_STRENGTH,
        "bump_range": LINE_IRREGULAR_BUMP_STRENGTH,
        "bump_count_range": LINE_IRREGULAR_BUMP_COUNT,
        "bump_width_frac_range": LINE_IRREGULAR_BUMP_WIDTH_FRAC,
        "bump_min_spacing_frac": LINE_IRREGULAR_BUMP_MIN_SPACING_FRAC,
        "noise_range": LINE_IRREGULAR_NOISE_STRENGTH,
        "morph_range": LINE_IRREGULAR_MORPH_STRENGTH,
    }
    for _ in range(LINE_DISTORT_PASSES):
        keep_band = distort_mask_irregular(keep_band, scale, **line_distort_kwargs)

    return keep_band


def generate_mask_by_method(method, img_shape, ratio_type="20-40"):
    """按指定方法生成破损遮罩"""
    if method == "ellipse":
        return generate_sain_ellipse_mask(img_shape, ratio_type=ratio_type)
    if method == "irregular":
        base_mask = generate_sain_ellipse_mask(img_shape, ratio_type=ratio_type)
        return distort_mask_irregular(base_mask, _image_scale(img_shape))
    if method == "line":
        return generate_line_damage_mask(img_shape)
    raise ValueError(f"未知遮罩方法: {method}")


def resolve_damage_mask(method, damage_mask, gray_mask, orig_leaf_area):
    """
    根据方法确定最终挖洞遮罩，并返回 (mask, retain_ratio, strategy_str)。
    damage_mask 中 255 表示需要移除（挖洞/涂白）的区域。

    直线法：保留直线带内叶片，移除带外区域；直线内叶片面积需 > MIN_RETAIN_RATIO。
    椭圆/不规则：沿用内外对抗保留策略。
    若当前方法无法得到有效遮罩，返回 (None, retain_ratio, reason)。
    """
    if orig_leaf_area == 0:
        return damage_mask, 0.0, "原面积为0"

    if method == "line":
        keep_band = damage_mask
        inside_ratio = np.sum((gray_mask > 0) & (keep_band > 0)) / orig_leaf_area
        if inside_ratio <= MIN_RETAIN_RATIO:
            return None, inside_ratio, f"直线内叶片 {inside_ratio*100:.1f}% ≤ {MIN_RETAIN_RATIO*100:.0f}%"
        if inside_ratio >= AREA_CHECK_THRESHOLD:
            return (
                None,
                inside_ratio,
                f"直线内叶片 {inside_ratio*100:.1f}% ≥ {AREA_CHECK_THRESHOLD*100:.0f}%",
            )
        remove_mask = np.where(keep_band > 0, 0, 255).astype(np.uint8)
        return remove_mask, inside_ratio, f"保留直线内 {inside_ratio*100:.1f}%"

    retain_ratio = np.sum((gray_mask > 0) & (damage_mask == 0)) / orig_leaf_area
    inside_ratio = 1.0 - retain_ratio

    if retain_ratio > MIN_RETAIN_RATIO and inside_ratio > MIN_RETAIN_RATIO:
        if inside_ratio < retain_ratio:
            damage_mask = 255 - damage_mask
            final_retain = inside_ratio
            strategy_str = "取较小[内]"
        else:
            final_retain = retain_ratio
            strategy_str = "取较小[外]"
    elif retain_ratio > MIN_RETAIN_RATIO:
        final_retain = retain_ratio
        strategy_str = "单边达标[外]"
    elif inside_ratio > MIN_RETAIN_RATIO:
        damage_mask = 255 - damage_mask
        final_retain = inside_ratio
        strategy_str = "单边达标[内]"
    else:
        return None, min(retain_ratio, inside_ratio), "内外侧均未达标"

    return damage_mask, final_retain, strategy_str


def try_generate_valid_mask(start_method_idx, img_shape, gray_mask, orig_leaf_area, ratio_type):
    """
    从起始方法起轮流尝试三种遮罩；每种方法最多 MAX_ATTEMPTS 次随机生成。
    返回 (damage_mask, final_retain, method, log_suffix)。
    """
    damage_mask = None
    final_retain = 1.0
    method_used = MASK_METHODS[start_method_idx % len(MASK_METHODS)]
    strategy_str = ""
    log_suffix = ""

    for method_offset in range(len(MASK_METHODS)):
        method = MASK_METHODS[(start_method_idx + method_offset) % len(MASK_METHODS)]
        method_name = MASK_METHOD_NAMES[method]

        for attempt in range(1, MAX_ATTEMPTS + 1):
            raw_mask = generate_mask_by_method(method, img_shape, ratio_type=ratio_type)
            resolved_mask, retain_ratio, strategy_str = resolve_damage_mask(
                method, raw_mask, gray_mask, orig_leaf_area
            )
            if resolved_mask is None:
                continue

            damage_mask = resolved_mask
            final_retain = retain_ratio
            method_used = method

            if final_retain < AREA_CHECK_THRESHOLD:
                log_suffix = (
                    f" [{method_name}] 第 {attempt} 次成功: {strategy_str}，"
                    f"保留 {final_retain*100:.1f}%"
                )
                break
        else:
            continue
        break
    else:
        if damage_mask is None:
            damage_mask = generate_mask_by_method(method_used, img_shape, ratio_type=ratio_type)
            damage_mask, final_retain, strategy_str = resolve_damage_mask(
                method_used, damage_mask, gray_mask, orig_leaf_area
            )
            if damage_mask is None:
                damage_mask = generate_sain_ellipse_mask(img_shape, ratio_type=ratio_type)
                damage_mask, final_retain, strategy_str = resolve_damage_mask(
                    "ellipse", damage_mask, gray_mask, orig_leaf_area
                )
        log_suffix = (
            f" [{MASK_METHOD_NAMES[method_used]}] 三种方法均未完全达标，"
            f"强制接受: {strategy_str}，保留 {final_retain*100:.1f}%"
        )

    return damage_mask, final_retain, method_used, log_suffix


def generate_sain_ellipse_mask(img_shape, ratio_type="20-40"):
    """根据 SAIN 论文策略生成控制随机性的椭圆遮罩 (Mask)"""
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    # 基于 256x256 基准分辨率计算缩放因子
    scale = _image_scale(img_shape)

    # 1. 约束：质心距离图像边界 30-50 像素
    border_min = int(30 * scale)
    border_max = int(50 * scale)
    if border_min >= W // 2 or border_min >= H // 2:
        border_min = min(W // 4, H // 4)

    cx0 = random.randint(border_min, W - border_min)
    cy0 = random.randint(border_min, H - border_min)
    centers = [(cx0, cy0)]

    # 2. 约束：随后的椭圆质心与核心质心距离不超过 20 像素
    max_dist = int(20 * scale)
    for _ in range(2):
        dx = random.randint(-max_dist, max_dist)
        dy = random.randint(-max_dist, max_dist)
        cx = np.clip(cx0 + dx, border_min, W - border_min)
        cy = np.clip(cy0 + dy, border_min, H - border_min)
        centers.append((int(cx), int(cy)))

    # 3. 绘制 3 个连通聚集的白色椭圆
    for cx, cy in centers:
        if ratio_type == "20-40":
            major_axis = int(random.randint(80, 90) * scale)
        elif ratio_type == "40-60":
            major_axis = int(random.randint(100, 110) * scale)
        else:  # "20-60" 广域随机模式
            major_axis = int(random.randint(80, 110) * scale)

        # 约束：长轴超过短轴 15-30 像素
        diff = int(random.randint(15, 30) * scale)
        minor_axis = max(10, major_axis - diff)
        angle = random.randint(0, 360)
        
        axes = (major_axis, minor_axis)

        # 填充绘制白色椭圆 (255)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

    return mask

def _load_required(path, label, filename):
    """读取必需的同名图片，失败时返回 (None, 错误信息)"""
    if not os.path.exists(path):
        return None, f"[-] 警告：在 {label} 中找不到对应图片，已跳过：{filename}"
    img = cv2.imread(path)
    if img is None:
        return None, f"[-] 错误：读取 {label}/{filename} 失败，已跳过"
    return img, None


def _find_fossil_jpg(filename):
    """在 leaf_fossil/diffusion/leaf 中查找同名 jpg（按文件名主干匹配）"""
    stem = os.path.splitext(filename)[0]
    fossil_jpg_path = os.path.join(FOSSIL_LEAF_DIR, f"{stem}.jpg")
    if os.path.isfile(fossil_jpg_path):
        return fossil_jpg_path
    return None


def _load_stone_background(parent_dir, filename, target_shape):
    """在父目录 stone/ 中查找同名背景图，尺寸不匹配时缩放到目标大小"""
    stone_dir = os.path.join(parent_dir, "stone")
    if not os.path.isdir(stone_dir):
        return None

    stem, ext = os.path.splitext(filename)
    candidates = [filename]
    for e in (ext, ".jpg", ".jpeg", ".png", ".webp"):
        name = f"{stem}{e}"
        if name not in candidates:
            candidates.append(name)

    for name in candidates:
        path = os.path.join(stone_dir, name)
        if not os.path.isfile(path):
            continue
        bg = cv2.imread(path)
        if bg is None:
            continue
        H, W = target_shape[:2]
        if bg.shape[0] != H or bg.shape[1] != W:
            bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
        return bg
    return None


def process_single_image(leaf_path, file_index=0):
    """单组图片的处理核心逻辑（多线程工作单元）"""
    try:
        leaf_dir, filename = os.path.split(leaf_path)
        parent_dir = os.path.dirname(leaf_dir)

        amodal_mask_dir = os.path.join(parent_dir, "amodal_mask")
        visible_mask_dir = os.path.join(parent_dir, "visible_mask")
        stone_mask_dir = os.path.join(parent_dir, "stone_mask")
        vein_mask_dir = os.path.join(parent_dir, "vein_mask")

        leaf_img = cv2.imread(leaf_path)
        if leaf_img is None:
            return False, f"[-] 错误：读取 leaf/{filename} 失败，已跳过"

        mask_img, err = _load_required(os.path.join(amodal_mask_dir, filename), "amodal_mask", filename)
        if err:
            return False, err
        visible_src, err = _load_required(os.path.join(visible_mask_dir, filename), "visible_mask", filename)
        if err:
            return False, err
        stone_img, err = _load_required(os.path.join(stone_mask_dir, filename), "stone_mask", filename)
        if err:
            return False, err
        vein_img, err = _load_required(os.path.join(vein_mask_dir, filename), "vein_mask", filename)
        if err:
            return False, err

        sync_fossil = False
        fossil_jpg_path = None
        if PROCESS_DIFFUSION:
            fossil_jpg_path = _find_fossil_jpg(filename)
            sync_fossil = fossil_jpg_path is not None
            if sync_fossil:
                os.makedirs(FOSSIL_OUT_DIR, exist_ok=True)
                shutil.copy2(leaf_path, os.path.join(FOSSIL_OUT_DIR, filename))

        gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) if len(mask_img.shape) == 3 else mask_img
        orig_leaf_area = np.sum(gray_mask > 0)
        log_messages = []
        fossil_suffix = " + 同步 leaf_fossil" if sync_fossil else ""

        for i in range(1, REPEAT_COUNT + 1):
            prefix = "broken_" if i == 1 else f"broken{i}_"
            start_method_idx = (file_index * REPEAT_COUNT + (i - 1)) % len(MASK_METHODS)

            if orig_leaf_area == 0:
                method_used = MASK_METHODS[start_method_idx]
                damage_mask = generate_mask_by_method(
                    method_used, leaf_img.shape, ratio_type=RATIO_TYPE
                )
                log_suffix = " (原面积为0，跳过校验)"
            else:
                damage_mask, final_retain, method_used, log_suffix = try_generate_valid_mask(
                    start_method_idx,
                    leaf_img.shape,
                    gray_mask,
                    orig_leaf_area,
                    RATIO_TYPE,
                )

            # leaf：挖洞区涂白，或 ellipse/irregular 以一定概率填充 stone 背景
            broken_leaf = leaf_img.copy()
            used_stone_bg = False
            if method_used in ("ellipse", "irregular") and random.random() < STONE_BG_PROB:
                stone_bg = _load_stone_background(parent_dir, filename, leaf_img.shape)
                if stone_bg is not None:
                    hole = damage_mask == 255
                    broken_leaf[hole] = stone_bg[hole]
                    used_stone_bg = True
                    log_suffix += " + stone背景"
            if not used_stone_bg:
                broken_leaf[damage_mask == 255] = [255, 255, 255]

            # stone 背景填充时不加阴影
            if not used_stone_bg and SHADOW_SIZE > 0:
                broken_gray = cv2.cvtColor(broken_leaf, cv2.COLOR_BGR2GRAY)
                object_mask = (damage_mask == 0) & (broken_gray < OBJECT_THRESHOLD)
                if np.any(object_mask):
                    bg_binary = np.ones(object_mask.shape, dtype=np.uint8)
                    bg_binary[object_mask] = 0
                    dist_map = cv2.distanceTransform(bg_binary, cv2.DIST_L2, 3)
                    shadow_filter = np.ones(dist_map.shape, dtype=np.float32)
                    shadow_zone = (dist_map > 0) & (dist_map <= SHADOW_SIZE)
                    shadow_filter[shadow_zone] = SHADOW_DARKNESS + (1.0 - SHADOW_DARKNESS) * (
                        dist_map[shadow_zone] / SHADOW_SIZE
                    )
                    for channel in range(3):
                        broken_leaf[:, :, channel] = np.clip(
                            broken_leaf[:, :, channel] * shadow_filter, 0, 255
                        ).astype(np.uint8)

            # visible / stone_mask：挖洞；amodal / vein：仅在开关开启时写 broken 副本
            broken_visible = visible_src.copy()
            broken_visible[damage_mask == 255] = 0
            broken_stone = stone_img.copy()
            broken_stone[damage_mask == 255] = 0

            broken_leaf_name = f"{prefix}{filename}"
            broken_leaf_path = os.path.join(leaf_dir, broken_leaf_name)
            cv2.imwrite(broken_leaf_path, broken_leaf)
            cv2.imwrite(os.path.join(visible_mask_dir, broken_leaf_name), broken_visible)
            cv2.imwrite(os.path.join(stone_mask_dir, broken_leaf_name), broken_stone)

            if COPY_UNDAMAGED_MASKS:
                cv2.imwrite(os.path.join(amodal_mask_dir, broken_leaf_name), mask_img)
                cv2.imwrite(os.path.join(vein_mask_dir, broken_leaf_name), vein_img)

            if sync_fossil:
                shutil.copy2(broken_leaf_path, os.path.join(FOSSIL_OUT_DIR, broken_leaf_name))
                stem = os.path.splitext(filename)[0]
                broken_jpg_name = f"{prefix}{stem}.jpg"
                shutil.copy2(fossil_jpg_path, os.path.join(FOSSIL_LEAF_DIR, broken_jpg_name))

            if SAVE_ELLIPSE_MASK:
                cv2.imwrite(
                    os.path.join(amodal_mask_dir, f"ellipse_{prefix}{filename}"),
                    damage_mask,
                )

            log_messages.append(f"   └─ {broken_leaf_name}{log_suffix}")

        final_log = (
            f"[+] 成功处理图片：{filename} (共生成 {REPEAT_COUNT} 张){fossil_suffix}\n"
            + "\n".join(log_messages)
        )
        return True, final_log

    except Exception as e:
        return False, f"[-] 异常错误：处理 {os.path.basename(leaf_path)} 时发生未预料的错误：{str(e)}"


def main():
    print("【SAIN 批量并发全掩码增强器-三方法轮换版】正在打开文件窗口...")
    method_cycle = " → ".join(MASK_METHOD_NAMES[m] for m in MASK_METHODS)
    print(
        f"配置模式: {RATIO_TYPE} 遮罩 | 方法轮换: {method_cycle} | "
        f"单图扩增: {REPEAT_COUNT} 张 | 有效面积目标: <{AREA_CHECK_THRESHOLD*100}% "
        f"(每种方法最多重试{MAX_ATTEMPTS}次，不达标自动切换)"
    )
    print(
        f"开关: COPY_UNDAMAGED_MASKS={COPY_UNDAMAGED_MASKS} | "
        f"PROCESS_DIFFUSION={PROCESS_DIFFUSION} | STONE_BG_PROB={STONE_BG_PROB}"
    )

    root = tk.Tk()
    root.withdraw()

    leaf_paths = filedialog.askopenfilenames(
        title="请批量选择 leaf 文件夹中的图片",
        filetypes=[
            ("Image Files", "*.png;*.jpg;*.jpeg;*.webp"),
        ]
    )

    if not leaf_paths:
        print("[-] 未选择任何文件，程序自动退出。")
        return

    total_files = len(leaf_paths)
    success_count = 0
    
    print(f"[+] 已选择 {total_files} 张 leaf 图片。正在启动多线程并发处理...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_path = {
            executor.submit(process_single_image, path, idx): path
            for idx, path in enumerate(leaf_paths)
        }
        
        for future in concurrent.futures.as_completed(future_to_path):
            success, message = future.result()
            if success:
                success_count += 1
            print(message)

    print("\n==================================================")
    print(f"[+] 批量并发处理完成！")
    print(f"[+] 成功处理 {success_count}/{total_files} 组原始图像，共计同步扩增生成 {success_count * REPEAT_COUNT} 组多模态掩码。")
    print("==================================================")

if __name__ == "__main__":
    main()