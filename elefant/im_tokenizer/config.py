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