from typing import Optional
import pydantic
from elefant.config import ConfigBase


class ConvTokenizerConfig(ConfigBase):
    num_tokens: int = 1


class VitTokenizerConfig(ConfigBase):
    patch_size: int = 16


class VjepaTokenizerConfig(ConfigBase):
    """
    V-JEPA 2 Tokenizer 配置类
    
    用于配置 V-JEPA 2 视觉编码器的各项参数。
    V-JEPA 2 是 Meta 开发的视频预训练模型，通过时空 patch 编码来提取视频特征。
    """
    # 本地 checkpoint 文件路径（优先使用）
    # 如果设置了此路径且文件存在，将从此路径加载模型权重
    # 如果为 None 或文件不存在，将根据 use_hub_fallback 决定是否从 torch.hub 下载
    checkpoint_path: Optional[str] = None
    
    # V-JEPA 2 模型架构名称
    # 可选值（V-JEPA 2.0）：
    #   - "vit_large": ViT-Large 模型（约 300M 参数，embed_dim=1024）
    #   - "vit_huge": ViT-Huge 模型（约 600M 参数，embed_dim=1280）
    #   - "vit_giant": ViT-Giant 模型（约 1.1B 参数，embed_dim=1408）
    #   - "vit_giant_384": ViT-Giant-384 模型（输入尺寸 384x384）
    # 可选值（V-JEPA 2.1）：
    #   - "vjepa2_1_vit_base_384"
    #   - "vjepa2_1_vit_large_384"
    model_name: str = "vit_large"
    
    # 是否冻结 V-JEPA 2 的参数（不进行梯度更新）
    # 推荐设置为 True，只训练投影层和 policy transformer
    # 如果设置为 False，将进行端到端微调（需要更多显存和计算资源）
    frozen: bool = True
    
    # V-JEPA 2 期望的输入图像尺寸（高度和宽度）
    # ViT-L/H/g 使用 256x256，vit_giant_384 使用 384x384
    # 如果实际输入尺寸不同，会自动进行 bilinear 插值 resize
    img_size: int = 256
    
    # 空间维度的 patch 大小（像素）
    # V-JEPA 2 将图像分割成 patch_size x patch_size 的 patches
    # 例如：img_size=256, patch_size=16 -> 16x16 = 256 个空间 patches
    patch_size: int = 16
    
    # 时间维度的 tubelet 大小（帧数）
    # V-JEPA 2 将连续 tubelet_size 帧压缩成一个时间步
    # 例如：tubelet_size=2 表示每 2 帧压缩成 1 个时间步
    # 这意味着输入 T 帧，输出 T//tubelet_size 个时间步的特征
    # 注意：V-JEPA 2 要求输入帧数 T >= tubelet_size（通常是 2）
    tubelet_size: int = 2
    
    # 预训练时使用的帧数（仅用于模型初始化，不影响实际输入）
    # V-JEPA 2 在预训练时使用 64 帧，但实际使用时可以输入任意帧数
    # 此参数主要用于正确初始化模型结构
    num_frames: int = 64
    
    # 当本地 checkpoint 不存在时，是否使用 torch.hub 作为后备方案
    # 如果设置为 True，将从 torch.hub 下载预训练权重（需要网络连接）
    # 如果设置为 False，本地 checkpoint 不存在时会抛出错误
    use_hub_fallback: bool = False

    # checkpoint 中 encoder 权重的 key
    # - "auto": 自动检测（优先 ema_encoder，回退 encoder/target_encoder）
    # - "ema_encoder": V-JEPA 2.1 常见 key
    # - "encoder": V-JEPA 2.0 常见 key
    checkpoint_key: str = "auto"

    # 是否将每帧 patch tokens 聚合为单个全局 token。
    # 开启后可显著降低 policy transformer 序列长度，缓解 OOM。
    pool_to_global: bool = False

    # 全局聚合后 MLP 的隐藏维度。None 时默认使用 2 * embed_dim。
    aggregation_mlp_hidden_dim: Optional[int] = None


class StaMoTokenizerConfig(ConfigBase):
    """
    StaMo Tokenizer 配置类

    StaMo 使用 timm ViT backbone + Q-Former 式 Projector 将单帧图像压缩为极少量 token。
    论文: "StaMo: Unsupervised Learning of Generalizable Robotic Motions from Static Images"
    """
    # timm ViT backbone 模型名称
    model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m"

    # 本地 backbone 权重路径（如果为 None 则使用 timm 在线下载）
    backbone_local_ckpt: Optional[str] = None

    # 是否使用 timm pretrained 权重（当 backbone_local_ckpt 为 None 时生效）
    backbone_pretrained: bool = True

    # Projector checkpoint 路径（包含训练好的 Projector 权重）
    # 如果为 None，则 Projector 使用随机初始化（仅用于测试）
    projector_ckpt: Optional[str] = None

    # 输入图像尺寸
    img_size: int = 224

    # 是否冻结全部参数（backbone + projector）
    frozen: bool = True

    # ---- Projector 超参 ----
    # 压缩后的 token 数量（StaMo 默认 2）
    num_token: int = 2

    # Projector 中 self+cross attn 层数
    num_attn_layers: int = 6

    # Projector 中逐级压缩 attn 层数
    num_attn_compress_layers: int = 6

    # Projector 内部隐藏维度
    hidden_dim: int = 1024

    # Projector cross-attention 维度
    cross_attention_dim: int = 512

    # Projector 输出对齐维度（最终每个 token 的特征维度）
    output_align_dim: int = 4096


class NitrogenSiglipTokenizerConfig(ConfigBase):
    """
    NitroGen 风格的 SigLIP 视觉编码器配置。

    不做 pooling，直接保留 dense visual tokens（例如 256 tokens/frame）。
    """

    # 视觉编码器名称（默认与 NitroGen 常用配置一致）
    vision_encoder_name: str = "google/siglip2-large-patch16-256"

    # 本地视觉编码器目录（优先级高于 vision_encoder_name）。
    # 例如: "./checkpoints/siglip2-large-patch16-256"
    vision_encoder_local_path: Optional[str] = None

    # 目标输入尺寸；若与实际输入不同，会先 resize 到该分辨率
    image_size: int = 256

    # 是否冻结视觉塔参数
    frozen: bool = True

    # 是否按 image processor 的 mean/std 做归一化
    use_image_processor_norm: bool = True

    # 期望每帧视觉 token 数（用于运行时一致性检查）
    expected_n_img_tokens: Optional[int] = 256

    # 期望视觉隐藏维度（用于运行时一致性检查，None 表示不检查）
    vision_hidden_size: Optional[int] = None


class NitrogenCheckpointTokenizerConfig(ConfigBase):
    """
    NitroGen checkpoint 对齐版视觉编码器配置。

    编码路径：
    image -> vision_encoder(SigLIP2) -> vl_self_attention_model(4-layer Transformer)
    """

    # NitroGen checkpoint 路径（默认使用仓库内路径）
    checkpoint_path: str = "NitroGen_checkpoints/ng.pt"

    # 视觉编码器名称（用于构建与 checkpoint 对齐的 backbone）
    vision_encoder_name: str = "google/siglip2-large-patch16-256"

    # 可选本地视觉编码器目录（优先级高于 vision_encoder_name）
    vision_encoder_local_path: Optional[str] = None

    # 模型输入图像尺寸
    image_size: int = 256

    # 是否按 image processor 的 mean/std 归一化
    use_image_processor_norm: bool = True

    # 是否冻结视觉塔和 VL mixing 模块
    freeze_vision_encoder: bool = True
    freeze_vl_self_attention_model: bool = True

    # 运行时一致性检查（None 表示跳过检查）
    expected_n_img_tokens: Optional[int] = 256
    vision_hidden_size: Optional[int] = 1024

    # vl_self_attention_model 结构（按 NitroGen 默认使用 4 层）
    vl_num_layers: int = 4
    vl_num_attention_heads: int = 16
    vl_attention_head_dim: int = 64
    vl_dropout: float = 0.1
    vl_attention_bias: bool = True
    vl_activation_fn: str = "gelu-approximate"
    vl_upcast_attention: bool = False
    vl_max_num_positional_embeddings: int = 512
    vl_compute_dtype: str = "float32"
    vl_final_dropout: bool = True
    vl_positional_embeddings: Optional[str] = "sinusoidal"

    # state_dict 加载策略
    strict_load: bool = False


class ImageTokenizerConfig(ConfigBase):
    type: str = "vit"
    conv_tokenizer_config: Optional[ConvTokenizerConfig] = pydantic.Field(
        default=ConvTokenizerConfig()
    )
    vit_tokenizer_config: Optional[VitTokenizerConfig] = pydantic.Field(
        default=VitTokenizerConfig()
    )
    vjepa_tokenizer_config: Optional[VjepaTokenizerConfig] = pydantic.Field(
        default=None
    )
    stamo_tokenizer_config: Optional[StaMoTokenizerConfig] = pydantic.Field(
        default=None
    )
    nitrogen_siglip_tokenizer_config: Optional[NitrogenSiglipTokenizerConfig] = pydantic.Field(
        default=None
    )
    nitrogen_checkpoint_tokenizer_config: Optional[NitrogenCheckpointTokenizerConfig] = pydantic.Field(
        default=None
    )