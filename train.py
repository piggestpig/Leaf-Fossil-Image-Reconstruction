"""直接运行此脚本训练AmodalDINO。

这是一个跨平台的精简版示例, 使用torch.accelerator.current_accelerator()自动识别GPU
下方有一些超参数可以调整, 比如
- BATCH_SIZE=8, 使用GRAD_CHECKPOINTING, 需要12G显存
- BATCH_SIZE=8, 关闭GRAD_CHECKPOINTING, 需要16G显存
- 论文中正式训练使用BATCH_SIZE=64, 需要96G显存 (RTX Pro 6000)
旁边的dataset文件夹放了1%的数据集, 供快速测试使用（训练完能初见成效）

如果要复现论文结果, 去github分支里下载完整数据集+服务器训练脚本, 
显存不够时, 把BATCH_SIZE改小, 再用GRAD_ACCUM_STEPS模拟大batch
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import LeafFossilDataset, collect_samples
from model import (
    AMODAL,
    AMODAL_VEIN,
    DETAIL_VEIN,
    IMAGE_SIZE,
    VISIBLE,
    build_model,
    pick_device,
)

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "dataset"
SAVE_PATH = HERE / "outputs" / "checkpoint.pt"

EPOCHS = 40
BATCH_SIZE = 8              # 论文中正式训练的batchsize=64, 需要96G显存
GRAD_ACCUM_STEPS = 1        # batchsize较低时, 可以增大此值, 使用梯度累计 通过多步模拟大batch
GRAD_CHECKPOINTING = True   # 节约显存, 反向传播时重算ViT block, 速度变慢
DECODER_LR = 1e-4
BACKBONE_LR = 1e-5          # DINOv3非常精细, 学习率太大会炸；余弦退火一段时间后才能学到叶脉
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0             # 启动子进程加载数据集（Windows开进程远比Linux慢, 要重新导入模块）
PRINT_EVERY = 5             # 每5步打印一次损失, 换大数据集不要这么做, 很吵

DEVICE = pick_device()
USE_BF16 = DEVICE.type != "cpu"


def boundary_weights(target: torch.Tensor) -> torch.Tensor:
    """w = 1 + 2T + 4∂T, 让loss更关注叶片内部和轮廓"""
    dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
    edge = (dilated - eroded).clamp(0.0, 1.0)
    return 1.0 + 2.0 * target + 4.0 * edge


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    weight = boundary_weights(target)
    if extra is not None:
        weight = weight * extra
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = 2.0 * (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((inter + eps) / (denom + eps)).mean()


def soft_tversky(logits: torch.Tensor, target: torch.Tensor, beta_fp = 0.65, beta_fn = 0.35, eps = 1.0) -> torch.Tensor:
    """Tversky是加权的Dice: β_FP > β_FN, 惩罚假阳性, 不要补出去一大团"""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)
    return 1.0 - ((tp + eps) / (tp + beta_fp * fp + beta_fn * fn + eps)).mean()


def area_overflow(probs: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    dims = (1, 2, 3)
    ratio = (probs.sum(dim=dims) / (target.sum(dim=dims) + eps) - 1.0).clamp(min=0.0)
    return ratio.pow(2.0).mean()


def head_loss_standard(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return weighted_bce(logits, target) + soft_dice(logits, target)


def head_loss_completion(logits: torch.Tensor, target: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    loss = weighted_bce(logits, target, extra) + soft_tversky(logits, target)
    loss = loss + 0.5 * area_overflow(probs, target)
    loss = loss + 0.25 * (probs * (1.0 - target)).mean()
    return loss


def total_loss(logits, visible, amodal, vein, detail) -> torch.Tensor:
    vis_l = logits[:, VISIBLE : VISIBLE + 1]
    amo_l = logits[:, AMODAL : AMODAL + 1]
    vein_l = logits[:, AMODAL_VEIN : AMODAL_VEIN + 1]
    det_l = logits[:, DETAIL_VEIN : DETAIL_VEIN + 1]

    # 补全区域加权
    completion = ((amodal > 0.5) & (visible < 0.5)).float()
    amodal_w = 1.0 + 2.0 * completion
    vein_w = 1.0 + 2.0 * ((vein > 0.5) & (visible < 0.5)).float()

    vis_p = torch.sigmoid(vis_l).detach()
    amo_p = torch.sigmoid(amo_l)
    vein_p = torch.sigmoid(vein_l)
    det_p = torch.sigmoid(det_l)

    # 结构约束: 可见叶片 / 叶脉应该落在完整叶片里面
    contain_vis = (vis_p * (1.0 - amo_p)).mean()
    contain_vein = (vein_p * (1.0 - amo_p.detach())).mean()
    contain_det = (det_p * (1.0 - amo_p.detach())).mean()

    return (
        1.0 * head_loss_standard(vis_l, visible)
        + 1.0 * head_loss_completion(amo_l, amodal, amodal_w)
        + 1.0 * head_loss_completion(vein_l, vein, vein_w)
        + 0.4 * head_loss_standard(det_l, detail)  # 细节叶脉质量不高, 权重小一点
        + 0.5 * contain_vis
        + 0.5 * contain_vein
        + 0.3 * contain_det
    )


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    dims = (1, 2, 3)
    inter = 2.0 * (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return float(((inter + eps) / (denom + eps)).mean().item())


def move_batch(leaf, roi, visible, amodal, vein, detail):
    leaf = leaf.to(DEVICE, non_blocking=True)
    roi = roi.to(DEVICE, non_blocking=True)
    if roi.ndim == 3:
        roi = roi.unsqueeze(1)
    x = torch.cat([leaf, roi], dim=1)
    visible = visible.to(DEVICE, non_blocking=True)
    amodal = amodal.to(DEVICE, non_blocking=True)
    vein = vein.to(DEVICE, non_blocking=True)
    detail = detail.to(DEVICE, non_blocking=True)
    return x, visible, amodal, vein, detail


@torch.inference_mode()
def evaluate(model, loader) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "visible": 0.0, "amodal": 0.0, "amodal_vein": 0.0, "detail_vein": 0.0}
    n = 0
    for leaf, roi, visible, amodal, vein, detail in loader:
        x, visible, amodal, vein, detail = move_batch(leaf, roi, visible, amodal, vein, detail)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(x)
            loss = total_loss(logits, visible, amodal, vein, detail)
        b = x.size(0)
        totals["loss"] += loss.item() * b
        totals["visible"] += dice_score(logits[:, VISIBLE : VISIBLE + 1], visible) * b
        totals["amodal"] += dice_score(logits[:, AMODAL : AMODAL + 1], amodal) * b
        totals["amodal_vein"] += dice_score(logits[:, AMODAL_VEIN : AMODAL_VEIN + 1], vein) * b
        totals["detail_vein"] += dice_score(logits[:, DETAIL_VEIN : DETAIL_VEIN + 1], detail) * b
        n += b
    return {key: value / max(n, 1) for key, value in totals.items()}


def main():
    print(f"device: {DEVICE}")
    train_set = LeafFossilDataset(collect_samples(DATA_ROOT / "train"), IMAGE_SIZE, augment=True)
    val_set = LeafFossilDataset(collect_samples(DATA_ROOT / "val"), IMAGE_SIZE, augment=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(load_dino_pretrain=True).to(DEVICE)

    if GRAD_CHECKPOINTING:
        from torch.utils.checkpoint import checkpoint
        for block in model.encoder.backbone.blocks:
            inner = block.forward
            def _ckpt_forward(*args, _inner=inner, **kwargs):
                return checkpoint(_inner, *args, use_reentrant=False, **kwargs)
            block.forward = _ckpt_forward

    # DINOv3 backbone 需要小学习率, decoder/heads可以使劲学
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("encoder.backbone") and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.backbone") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_params, "lr": DECODER_LR},
            {"params": backbone_params, "lr": BACKBONE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_amodal = -1.0
    assert GRAD_ACCUM_STEPS >= 1, "GRAD_ACCUM_STEPS 必须 >= 1"
    print(f"train={len(train_set)}  val={len(val_set)}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"batch={BATCH_SIZE}  accum={GRAD_ACCUM_STEPS}  effective_batch={BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"grad_checkpointing: {GRAD_CHECKPOINTING}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        total_steps = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        for step, (leaf, roi, visible, amodal, vein, detail) in enumerate(train_loader, start=1):
            x, visible, amodal, vein, detail = move_batch(leaf, roi, visible, amodal, vein, detail)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(x)
                loss = total_loss(logits, visible, amodal, vein, detail)

            # 梯度累计模拟大batch
            (loss / GRAD_ACCUM_STEPS).backward()
            should_step = step % GRAD_ACCUM_STEPS == 0 or step == total_steps
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * x.size(0)
            seen += x.size(0)
            if step == 1 or step % PRINT_EVERY == 0 or step == total_steps:
                print(f"epoch {epoch}/{EPOCHS}  step {step}/{total_steps}  loss={loss.detach().item():.4f}")

        val = evaluate(model, val_loader)
        scheduler.step()
        print(
            f"epoch {epoch}/{EPOCHS}  train_loss={running / max(seen, 1):.4f}  "
            f"val_loss={val['loss']:.4f}  val_amodal_dice={val['amodal']:.4f}  "
            f"vein={val['amodal_vein']:.4f}  detail={val['detail_vein']:.4f}"
        )
        if val["amodal"] > best_amodal:
            best_amodal = val["amodal"]
            torch.save(
                {
                    "model_kind": "amodal_dino",
                    "backbone_name": "vitl16",
                    "image_size": IMAGE_SIZE,
                    "input_channels": 4,
                    "out_channels": 4,
                    "model_config": {
                        "architecture": "dino_dpt",
                        "backbone_name": "vitl16",
                        "weights_path": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                        "frozen_backbone": False,
                        "inject_roi_in_backbone": True,
                        "in_channels": 4,
                        "out_channels": 4,
                        "trunk_channels": 256,
                        "use_backbone_norm": True,
                        "patch_divisor": 16,
                        "unfreeze_last_n_blocks": 0,
                    },
                    "model_state": model.state_dict(),
                },
                SAVE_PATH,
            )
            print(f"saved best (amodal dice={best_amodal:.4f}): {SAVE_PATH}")


if __name__ == "__main__":
    main()
