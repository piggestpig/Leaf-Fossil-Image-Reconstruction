"""AmodalDINO：RGB+ROI → 四个分割头。

数据流 (对应论文 Method 一节) ：

    RGB(3) + ROI(1)
        → ImageNet / 0.5 归一化
        → 4 通道 patch embedding (DINOv3 ViT-L/16, 全程微调) 
        → 取出第 4/11/17/23 个 block 的 token
        → 官方 DPT trunk 融合成 256 通道特征图
        → 四个独立 3×3 卷积头
        → visible / amodal / amodal_vein / detail_vein
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dinov3"))

from dinov3.eval.depth.models.dpt_head import DPTHead
from dinov3.hub.backbones import dinov3_vitl16

# 四个输出头的通道顺序, 训练和推理必须一致
VISIBLE, AMODAL, AMODAL_VEIN, DETAIL_VEIN = 0, 1, 2, 3
HEAD_NAMES = ("visible", "amodal", "amodal_vein", "detail_vein")

IMAGE_SIZE = 448
PATCH_SIZE = 16
EMBED_DIM = 1024
TRUNK_CHANNELS = 256
# ViT-L 共 24 层, 官方 DPT 取均匀四层；ViT-L 上固定为 [4, 11, 17, 23]
FEATURE_BLOCKS = (4, 11, 17, 23)

# ImageNet 均值方差；ROI 用 0.5/0.5, 这样中性提示 0.5 归一化后正好是 0
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

DINO_WEIGHTS = HERE / "weights" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
CHECKPOINT = HERE / "weights" / "amodal_dino_vitl16_roi_full_indep.pt"


def pick_device() -> torch.device:
    """优先使用当前机器上的加速器 (CUDA / MPS / XPU), 没有就退回 CPU。"""
    device = torch.accelerator.current_accelerator()
    return device if device is not None else torch.device("cpu")


def expand_patch_embed_to_4ch(backbone: nn.Module) -> None:
    """把 patch embedding 从 3 通道扩成 4 通道, 并保留预训练的 RGB 卷积核。ROI取RGB均值, 中性ROI = 0.5
    """
    old = backbone.patch_embed.proj
    if old.in_channels == 4:
        return
    new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
        new.weight[:, 3:4].copy_(old.weight.mean(dim=1, keepdim=True))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    backbone.patch_embed.proj = new
    backbone.patch_embed.in_chans = 4


class BackboneEncoder(nn.Module):
    """包一层 DINOv3, 取出 DPT 需要的四组 (空间特征, cls token)。"""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.backbone.get_intermediate_layers(
            x,
            n=FEATURE_BLOCKS,
            reshape=True,
            return_class_token=True,
            norm=True,
        )


class DPTTrunk(nn.Module):
    """官方 DPT 的 Reassemble + Fusion, 输出 256 通道、半分辨率特征图。"""

    def __init__(self):
        super().__init__()
        self.dpt = DPTHead(
            in_channels=[EMBED_DIM] * 4,
            channels=TRUNK_CHANNELS,
            post_process_channels=[128, 256, 512, 768],
            readout_type="project",
            use_batchnorm=False,
            use_bias=False,
            n_output_channels=TRUNK_CHANNELS,
            n_hidden_channels=32,
        )

    def forward(self, features: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.dpt.forward_features(features)


class FourHeads(nn.Module):
    """四个彼此独立的 3×3 卷积, 不共享权重, 方便每个任务自己学。"""

    def __init__(self):
        super().__init__()
        self.head_visible = nn.Conv2d(TRUNK_CHANNELS, 1, kernel_size=3, padding=1)
        self.head_amodal = nn.Conv2d(TRUNK_CHANNELS, 1, kernel_size=3, padding=1)
        self.head_vein = nn.Conv2d(TRUNK_CHANNELS, 1, kernel_size=3, padding=1)
        self.head_detail_vein = nn.Conv2d(TRUNK_CHANNELS, 1, kernel_size=3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.head_visible(features),
                self.head_amodal(features),
                self.head_vein(features),
                self.head_detail_vein(features),
            ],
            dim=1,
        )


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = DPTTrunk()
        self.heads = FourHeads()


class AmodalDINO(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.encoder = BackboneEncoder(backbone)
        self.decoder = Decoder()
        self.register_buffer("rgb_mean", torch.tensor(RGB_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor(RGB_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, H, W), 前 3 通道 RGB ∈ [0,1], 第 4 通道 ROI ∈ {0.5, 1}。"""
        rgb = (x[:, :3] - self.rgb_mean) / self.rgb_std
        roi = (x[:, 3:] - 0.5) / 0.5

        _, _, height, width = rgb.shape
        pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        if pad_h or pad_w:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="reflect")
            roi = F.pad(roi, (0, pad_w, 0, pad_h), mode="reflect")

        features = self.encoder(torch.cat([rgb, roi], dim=1))
        logits = self.decoder.heads(self.decoder.trunk(features))
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return logits[..., :height, :width]


def build_model(load_dino_pretrain: bool = True) -> AmodalDINO:
    """新建模型。训练时加载 DINOv3 预训练；推理时随后会被 checkpoint 覆盖, 可跳过。"""
    backbone = dinov3_vitl16(pretrained=False)
    if load_dino_pretrain:
        if not DINO_WEIGHTS.is_file():
            raise FileNotFoundError(f"找不到 DINOv3 预训练权重: {DINO_WEIGHTS}")
        state = torch.load(DINO_WEIGHTS, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state)
    expand_patch_embed_to_4ch(backbone)
    return AmodalDINO(backbone)


def load_trained_model(checkpoint_path: Path | None = None, device: torch.device | None = None) -> tuple[AmodalDINO, int]:
    """加载论文发布的微调权重, 返回 (模型, 训练时的输入边长)。"""
    device = device or pick_device()
    path = checkpoint_path or CHECKPOINT
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(load_dino_pretrain=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, int(ckpt.get("image_size", IMAGE_SIZE))
