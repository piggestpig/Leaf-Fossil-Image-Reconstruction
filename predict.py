"""用训练完的checkpoint或预训练权重对一张图/一个文件夹进行推理。
可传入两个参数: 图片路径、模型路径, 例如
python predict.py
python predict.py dataset/val/leaf_fossil/broken_2001.jpg
python predict.py dataset/val/leaf_fossil
python predict.py dataset/test
python predict.py dataset/test  weights/amodal_dino_vitl16_roi_full_indep.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torchvision.io import ImageReadMode

from model import CHECKPOINT, IMAGE_SIZE, load_trained_model, pick_device

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "dataset" / "val" / "leaf_fossil"
TUTORIAL_CHECKPOINT = HERE / "outputs" / "checkpoint.pt"
OUTPUT_DIR = HERE / "outputs"
THRESHOLD = 0.5
MAX_LONG_EDGE = 1024
OVERLAY_ALPHA = 0.45

# 叠加图：可见=绿，完整叶片=浅黄，主脉=品红，细脉=紫
HEAD_COLORS = {
    "visible": (0, 200, 0),
    "amodal": (220, 210, 70),
    "amodal_vein": (220, 0, 160),
    "detail_vein": (140, 40, 180), 
}


def collect_images(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    raise FileNotFoundError(path)


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    rgb = torchvision.io.read_image(str(path), mode=ImageReadMode.RGB).to(device=device, dtype=torch.float32) / 255.0
    h, w = rgb.shape[-2:]
    long_edge = max(h, w)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        rgb = F.interpolate(rgb.unsqueeze(0), size=(max(1, round(h * scale)), max(1, round(w * scale))), mode="bilinear", align_corners=False)[0]
    return rgb


def overlay(rgb: torch.Tensor, masks: dict[str, torch.Tensor]) -> torch.Tensor:
    out = rgb.clone()
    for name, color in HEAD_COLORS.items():
        color = torch.tensor(color, device=rgb.device, dtype=rgb.dtype).view(3, 1, 1) / 255.0
        m = masks[name].float()
        out = out * (1.0 - OVERLAY_ALPHA * m) + color * (OVERLAY_ALPHA * m)
    return out.clamp(0.0, 1.0)


def save_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.io.write_png((image.clamp(0, 1) * 255).byte().cpu(), str(path))


def resolve_checkpoint() -> Path:
    if len(sys.argv) > 2:
        return Path(sys.argv[2])
    if TUTORIAL_CHECKPOINT.is_file():
        return TUTORIAL_CHECKPOINT
    return CHECKPOINT


def resolve_output_dir(input_path: Path) -> Path:
    text = str(input_path).replace("/", "/").lower()
    if "/test" in text or input_path.name == "test":
        return OUTPUT_DIR / "test"
    if "/val" in text:
        return OUTPUT_DIR / "val"
    return OUTPUT_DIR


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    checkpoint = resolve_checkpoint()
    output_dir = resolve_output_dir(input_path)
    device = pick_device()
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint}")
    print(f"output: {output_dir}")

    model, image_size = load_trained_model(checkpoint_path=checkpoint, device=device)
    image_size = image_size or IMAGE_SIZE
    paths = collect_images(input_path)
    if not paths:
        raise FileNotFoundError(f"没有图片: {input_path}")

    with torch.inference_mode():
        for path in paths:
            rgb = load_rgb(path, device)
            rgb_in = F.interpolate(rgb.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
            # 直接用中性ROI=0.5, 实际上ROI是早期设计产物, 后面不怎么用了
            roi_in = torch.full((1, 1, image_size, image_size), 0.5, device=device, dtype=rgb_in.dtype)
            x = torch.cat([rgb_in, roi_in], dim=1)
            logits = model(x)[0]
            masks = {}
            for i, name in enumerate(HEAD_COLORS.keys()):
                prob = torch.sigmoid(logits[i])
                prob = F.interpolate(prob.unsqueeze(0).unsqueeze(0), size=rgb.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                masks[name] = prob > THRESHOLD
            vis = overlay(rgb, masks)
            out_path = output_dir / f"{path.stem}_pred.png"
            save_png(vis, out_path)
            print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
