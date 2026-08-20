"""加载叶片化石数据集
    dataset/train|val/
        leaf_fossil/          破损化石RGB (输入图像)
        visible_mask/         可见叶片
        amodal_mask/          完整叶片
        vein_mask/            主脉
        detail_vein_mask/     细脉
        stone_mask/           石头遮罩, 可选, 用作ROI提示的一部分
    dataset/test/             真实化石照片（无 GT, 只用于推理）

破损图文件名可能带 broken_ / broken2_ 等前缀, 完整叶片的 GT 往往只有 0001.jpg。
配对规则：先精确同名, 对不上就反复去掉第一个 ``前缀_``。
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ROI_SOURCES = ("stone", "visible", "amodal", "amodal_vein", "neutral")


def resolve_paired_path(directory: Path, leaf_name: str) -> Path | None:
    """精确同名；否则不断剥掉 ``xxx_`` 前缀再试。"""
    candidate = directory / leaf_name
    if candidate.is_file():
        return candidate
    name = leaf_name
    while "_" in name:
        name = name.split("_", 1)[1]
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def collect_samples(split_dir: Path) -> list[dict]:
    leaf_dir = split_dir / "leaf_fossil"
    folders = {
        "visible": split_dir / "visible_mask",
        "amodal": split_dir / "amodal_mask",
        "vein": split_dir / "vein_mask",
        "detail_vein": split_dir / "detail_vein_mask",
    }
    for name, path in {"leaf_fossil": leaf_dir, **folders}.items():
        if not path.is_dir():
            raise FileNotFoundError(f"缺少目录 {name}: {path}")

    stone_dir = split_dir / "stone_mask"
    samples = []
    for leaf_path in sorted(leaf_dir.rglob("*")):
        if not leaf_path.is_file() or leaf_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        paths = {key: resolve_paired_path(folder, leaf_path.name) for key, folder in folders.items()}
        if any(path is None for path in paths.values()):
            continue
        samples.append(
            {
                "leaf": leaf_path,
                **paths,
                "stone": resolve_paired_path(stone_dir, leaf_path.name) if stone_dir.is_dir() else None,
            }
        )
    if not samples:
        raise FileNotFoundError(f"{split_dir} 下没有配对成功的样本")
    return samples


def _read_rgb(path: Path, size: int) -> torch.Tensor:
    image = torchvision.io.read_image(str(path), mode=ImageReadMode.RGB).float()
    image = F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    return image.clamp(0, 255) / 255.0


def _read_mask(path: Path | None, size: int, like: torch.Tensor) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, size, size, dtype=torch.float32)
    mask = torchvision.io.read_image(str(path), mode=ImageReadMode.GRAY).float()
    mask = F.interpolate(mask.unsqueeze(0), size=(size, size), mode="nearest")[0]
    return (mask > 127).float()


def _dilate(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel, stride=1, padding=kernel // 2)[0]


def _erode(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - mask, kernel)


def _elastic_warp(mask: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
    """用平滑随机位移场扭曲ROI提示, 防止网络去描边GT轮廓。"""
    _, height, width = mask.shape
    if height < 8 or width < 8:
        return mask
    kernel = max(3, int(round(sigma * 4)) | 1)
    dx = v2.functional.gaussian_blur(torch.randn(1, height, width) * alpha, [kernel, kernel], [sigma, sigma])
    dy = v2.functional.gaussian_blur(torch.randn(1, height, width) * alpha, [kernel, kernel], [sigma, sigma])
    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = 2.0 * (grid_x + dx[0]) / max(width - 1, 1) - 1.0
    grid_y = 2.0 * (grid_y + dy[0]) / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    warped = F.grid_sample(mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=True)[0]
    return (warped > 0.5).float()


def corrupt_roi(roi: torch.Tensor, allow_erode: bool) -> torch.Tensor:
    """随机膨胀 / 腐蚀 + 弹性变形。不腐蚀的话模型会把提示当成答案来描。"""
    kernel = random.choice((3, 5, 7, 9))
    repeats = random.randint(1, 3)
    roll = random.random()
    if allow_erode:
        if roll < 0.40:
            for _ in range(repeats):
                roi = _dilate(roi, kernel)
        elif roll < 0.65:
            for _ in range(repeats):
                roi = _erode(roi, kernel)
        elif roll < 0.80:
            for _ in range(max(1, repeats - 1)):
                roi = _dilate(roi, kernel)
            roi = _erode(roi, kernel)
    elif roll < 0.85:
        # 主脉只有一两像素宽, 腐蚀会直接消失, 所以只允许膨胀
        for _ in range(repeats):
            roi = _dilate(roi, kernel)
    if random.random() < 0.9:
        roi = _elastic_warp(roi, alpha=random.uniform(6.0, 22.0), sigma=random.uniform(3.0, 7.0))
    return (roi > 0.5).float()


def encode_roi(binary: torch.Tensor) -> torch.Tensor:
    """中性ROI=0.5"""
    hint = (binary > 0.5).float()
    return torch.full_like(hint, 0.5) + 0.5 * hint


def sample_roi(stone, visible, amodal, vein, has_stone: bool, training: bool) -> torch.Tensor:
    """随机挑选ROI进行训练...其实ROI的设计已经不怎么用了"""
    if not training:
        return encode_roi(stone) if has_stone else torch.full_like(stone, 0.5)

    choice = random.choice(ROI_SOURCES)
    if choice == "stone" and has_stone:
        return encode_roi(corrupt_roi(stone, allow_erode=True))
    if choice == "visible":
        return encode_roi(corrupt_roi(visible, allow_erode=True))
    if choice == "amodal":
        return encode_roi(corrupt_roi(amodal, allow_erode=True))
    if choice == "amodal_vein":
        return encode_roi(corrupt_roi(vein, allow_erode=False))
    return torch.full_like(stone, 0.5)


def scale_about_point(image: torch.Tensor, scale: float, cx: float, cy: float, mode: str, fill: float) -> torch.Tensor:
    """非常重要！绕图像内随机点缩放, 否则模型学到的叶片都在中间。原图背景区域填白, mask背景区域填0"""
    _, height, width = image.shape
    if abs(scale - 1.0) < 1e-6:
        return image
    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    src_x = (grid_x - cx) / scale + cx
    src_y = (grid_y - cy) / scale + cy
    grid_x_n = 2.0 * src_x / max(width - 1, 1) - 1.0
    grid_y_n = 2.0 * src_y / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x_n, grid_y_n], dim=-1).unsqueeze(0)
    warped = F.grid_sample(image.unsqueeze(0), grid, mode=mode, padding_mode="zeros", align_corners=True)[0]
    outside = (src_x < 0) | (src_x > width - 1) | (src_y < 0) | (src_y > height - 1)
    return torch.where(outside.unsqueeze(0), torch.full_like(warped, fill), warped)


class LeafFossilDataset(Dataset):
    def __init__(self, samples: list[dict], image_size: int, augment: bool):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        # 化石本身偏暗, 通常调亮一点
        self.color_jitter = v2.ColorJitter(brightness=(0.75, 1.65), contrast=0.15, saturation=0.1, hue=0.05)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, leaf, visible, amodal, stone, vein, detail):
        tensors = [leaf, visible, amodal, stone, vein, detail]
        if random.random() < 0.5:
            tensors = [torch.flip(t, dims=[2]) for t in tensors]
        if random.random() < 0.5:
            tensors = [torch.flip(t, dims=[1]) for t in tensors]
        k = random.randint(0, 3)
        if k:
            tensors = [torch.rot90(t, k=k, dims=[1, 2]) for t in tensors]
        leaf, visible, amodal, stone, vein, detail = tensors

        _, height, width = leaf.shape
        scale = random.uniform(0.5, 1.0)
        cx = random.uniform(0.0, max(width - 1, 0))
        cy = random.uniform(0.0, max(height - 1, 0))
        leaf = scale_about_point(leaf, scale, cx, cy, mode="bilinear", fill=1.0)
        visible = scale_about_point(visible, scale, cx, cy, mode="nearest", fill=0.0)
        amodal = scale_about_point(amodal, scale, cx, cy, mode="nearest", fill=0.0)
        stone = scale_about_point(stone, scale, cx, cy, mode="nearest", fill=0.0)
        vein = scale_about_point(vein, scale, cx, cy, mode="nearest", fill=0.0)
        detail = scale_about_point(detail, scale, cx, cy, mode="nearest", fill=0.0)
        leaf = self.color_jitter(leaf)
        return leaf, visible, amodal, stone, vein, detail

    def __getitem__(self, index: int):
        sample = self.samples[index]
        leaf = _read_rgb(sample["leaf"], self.image_size)
        visible = _read_mask(sample["visible"], self.image_size, leaf)
        amodal = _read_mask(sample["amodal"], self.image_size, leaf)
        vein = _read_mask(sample["vein"], self.image_size, leaf)
        detail = _read_mask(sample["detail_vein"], self.image_size, leaf)
        stone = _read_mask(sample["stone"], self.image_size, leaf)
        has_stone = sample["stone"] is not None and stone.sum() > 0

        if self.augment:
            leaf, visible, amodal, stone, vein, detail = self._augment(leaf, visible, amodal, stone, vein, detail)

        roi = sample_roi(stone, visible, amodal, vein, has_stone, training=self.augment)
        return leaf, roi, visible, amodal, vein, detail
