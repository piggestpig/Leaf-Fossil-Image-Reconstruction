"""DINOv3 ViT-L/16 构造函数（从官方 hub 精简，只保留本教程用到的一种 backbone）。"""

from dinov3.models.vision_transformer import DinoVisionTransformer


def dinov3_vitl16(pretrained: bool = False, **kwargs):
    """构建 ViT-L/16：embed_dim=1024，24 个 block，patch=16。

    预训练权重由调用方用 ``torch.load`` 加载，这里默认 ``pretrained=False``，
    避免官方 hub 再去联网下载。
    """
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=False,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model
