"""
提取数据集中视频帧的 DINOv2 视觉特征，用于监督 PolicyFutureCausalTransformer 的 future_vision_head。

使用方法：
    python scripts/extract_future_vision_features.py \
        --dataset_dir dataset \
        --output_dir dataset_features \
        --embed_dim 256 \
        --batch_size 32 \
        --device cuda:0

输出：
    对于每个视频样本，生成一个 .pt 文件，包含该视频所有帧的 DINOv2 特征。
    文件格式：[T, 196, 768] - 时间步数, token数, DINOv2特征维度
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    VideoDecoder = None
    logging.warning("VideoDecoder not available. Please install torchcodec.")

from elefant.data.proto import video_annotation_pb2


class DinoV2FeatureExtractor:
    """
    使用 DINOv2 提取图像特征的提取器。
    
    与 DinoV2Tokenizer 类似，但不包含投影层，直接返回 768 维的 DINOv2 特征。
    """

    def __init__(self, device: str = "cuda:0"):
        """
        初始化 DINOv2 特征提取器。
        
        Args:
            device: 计算设备（"cuda:0" 或 "cpu"）
        """
        self.device = torch.device(device)
        
        # 加载 DINOv2 模型
        logging.info("Loading DINOv2 model from torch.hub...")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.to(self.device)
        self.model.eval()
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # DINOv2 配置
        self.n_image_tokens = 196  # 14x14 patches
        self.dino_embed_dim = 768  # ViT-Base 的 embedding 维度
        self.input_size = 192  # 输入图像尺寸
        self.padded_size = 196  # 填充后的尺寸
        
        logging.info(f"DINOv2 feature extractor initialized on {device}")

    def extract_global_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        提取图像的全局 DINOv2 特征（对 patch tokens 求平均）。
        
        用于监督 a⁰（全局动作决策 token），应该使用全局视觉表征。
        
        Args:
            frames: 输入图像张量，形状为 [B, T, C, H, W] 或 [T, C, H, W] 或 [B, C, H, W]
        
        Returns:
            全局特征张量，形状为 [B, T, 768] 或 [T, 768] 或 [B, 768]
                - 768: DINOv2 特征维度（全局特征）
        """
        # 先提取 patch tokens
        patch_tokens = self.extract_features(frames)  # [B, T, 196, 768] 或 [T, 196, 768] 或 [B, 196, 768]
        
        # 对空间维度（patch tokens 维度）求平均
        # 如果 patch_tokens 是 [B, T, 196, 768]，对 dim=2 求平均 -> [B, T, 768]
        # 如果 patch_tokens 是 [T, 196, 768]，对 dim=1 求平均 -> [T, 768]
        # 如果 patch_tokens 是 [B, 196, 768]，对 dim=1 求平均 -> [B, 768]
        if patch_tokens.dim() == 4:
            # [B, T, 196, 768]
            global_features = patch_tokens.mean(dim=2)  # [B, T, 768]
        elif patch_tokens.dim() == 3:
            # [T, 196, 768] 或 [B, 196, 768]
            global_features = patch_tokens.mean(dim=1)  # [T, 768] 或 [B, 768]
        else:
            raise ValueError(f"Unexpected patch_tokens dimension: {patch_tokens.dim()}")
        
        return global_features

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        提取图像帧的 DINOv2 特征。
        
        Args:
            frames: 输入图像张量，形状为 [B, T, C, H, W] 或 [T, C, H, W] 或 [B, C, H, W]
                - B: batch size（可选）
                - T: 时间步数（帧数，可选）
                - C: 通道数（3，RGB）
                - H: 高度（必须是 192）
                - W: 宽度（必须是 192）
        
        Returns:
            特征张量，形状与输入对应（去掉 C, H, W，添加 196, 768）
                - 196: patch tokens 数量
                - 768: DINOv2 特征维度
        """
        # 处理输入维度
        original_dim = frames.dim()
        original_shape = frames.shape
        
        if original_dim == 4:
            # [T, C, H, W] 或 [B, C, H, W]
            # 判断是时间维度还是 batch 维度
            if frames.shape[1] == 3:  # [B, C, H, W] - 第二个维度是通道
                # 这是单帧的 batch，添加时间维度
                frames = frames.unsqueeze(1)  # [B, 1, C, H, W]
                squeeze_time = True
                squeeze_batch = False
            else:  # [T, C, H, W] - 第一个维度是时间
                # 这是单 batch 的时间序列，添加 batch 维度
                frames = frames.unsqueeze(0)  # [1, T, C, H, W]
                squeeze_time = False
                squeeze_batch = True
        elif original_dim == 5:
            # [B, T, C, H, W]
            squeeze_time = False
            squeeze_batch = False
        else:
            raise ValueError(f"Unexpected input dimension: {original_dim}, shape: {original_shape}")
        
        B, T, C, H, W = frames.shape
        
        # 检查输入尺寸
        if H != self.input_size or W != self.input_size:
            raise ValueError(
                f"DINOv2 requires {self.input_size}x{self.input_size} images, "
                f"got {H}x{W}"
            )
        
        # 移动到设备
        frames = frames.to(self.device)
        
        with torch.no_grad():
            # 填充：将 192x192 填充到 196x196（左右上下各 2 像素）
            x = F.pad(frames, (2, 2, 2, 2), value=0.0)
            # x: [B, T, C, 196, 196]
            
            # 将 batch 和 time 维度合并
            x = x.view(B * T, C, self.padded_size, self.padded_size)
            
            # 调用 DINOv2 提取特征
            features = self.model.forward_features(x)
            # features 是一个字典，包含多个输出
            
            # 提取归一化的 patch tokens
            patch_tokens = features["x_norm_patchtokens"]
            # patch_tokens: [B*T, 196, 768]
            
            # 重新 reshape 回 [B, T, 196, 768]
            patch_tokens = patch_tokens.view(B, T, self.n_image_tokens, self.dino_embed_dim)
        
        # 恢复原始维度（如果需要）
        if squeeze_batch:
            patch_tokens = patch_tokens.squeeze(0)  # [T, 196, 768]
        elif squeeze_time:
            patch_tokens = patch_tokens.squeeze(1)  # [B, 196, 768]
        
        return patch_tokens


def process_video(
    video_path: Path,
    feature_extractor: DinoV2FeatureExtractor,
    batch_size: int = 32,
    use_global_features: bool = True,
) -> Optional[torch.Tensor]:
    """
    处理单个视频，提取所有帧的 DINOv2 特征。
    
    Args:
        video_path: 视频文件路径
        feature_extractor: DINOv2 特征提取器
        batch_size: 批处理大小（用于加速处理）
        use_global_features: 如果 True，返回全局特征（对 patch tokens 求平均）[T, 768]
                            如果 False，返回 patch tokens [T, 196, 768]
    
    Returns:
        特征张量，形状为 [T, 768]（全局特征）或 [T, 196, 768]（patch tokens），如果处理失败返回 None
    """
    try:
        # 使用 VideoDecoder 读取视频
        decoder = VideoDecoder(str(video_path), device="cpu", num_ffmpeg_threads=1)
        n_frames = len(decoder)
        
        if n_frames == 0:
            logging.warning(f"Video {video_path} has 0 frames, skipping.")
            return None
        
        # 读取所有帧
        all_frames = []
        for i in range(n_frames):
            frame = decoder[i]  # [C, H, W], uint8
            # 转换为 float32 并归一化到 [0, 1]
            frame = frame.float() / 255.0
            # 确保尺寸是 192x192（如果不是，需要 resize）
            if frame.shape[1] != 192 or frame.shape[2] != 192:
                frame = F.interpolate(
                    frame.unsqueeze(0),
                    size=(192, 192),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            all_frames.append(frame)
        
        # 堆叠所有帧: [T, C, H, W]
        frames_tensor = torch.stack(all_frames, dim=0)
        
        # 批量提取特征（加速处理）
        if use_global_features:
            # 提取全局特征（对 patch tokens 求平均）
            all_features = []
            for i in range(0, n_frames, batch_size):
                batch_frames = frames_tensor[i : i + batch_size]  # [batch_size, C, H, W]
                batch_features = feature_extractor.extract_global_features(batch_frames)  # [batch_size, 768]
                all_features.append(batch_features)
            
            # 拼接所有批次的特征
            features = torch.cat(all_features, dim=0)  # [T, 768]
        else:
            # 提取 patch tokens
            all_features = []
            for i in range(0, n_frames, batch_size):
                batch_frames = frames_tensor[i : i + batch_size]  # [batch_size, C, H, W]
                batch_features = feature_extractor.extract_features(batch_frames)  # [batch_size, 196, 768]
                all_features.append(batch_features)
            
            # 拼接所有批次的特征
            features = torch.cat(all_features, dim=0)  # [T, 196, 768]
        
        return features
        
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")
        return None


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    feature_extractor: DinoV2FeatureExtractor,
    batch_size: int = 32,
    sample_id: Optional[str] = None,
    use_global_features: bool = True,
):
    """
    处理数据集中的所有视频，提取 DINOv2 特征。
    
    Args:
        dataset_dir: 数据集根目录（包含多个样本目录）
        output_dir: 输出目录（保存特征文件）
        feature_extractor: DINOv2 特征提取器
        batch_size: 批处理大小
        sample_id: 如果指定，只处理该样本（用于测试）
    """
    if VideoDecoder is None:
        raise ImportError("VideoDecoder is not available. Please install torchcodec.")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有样本目录
    sample_dirs = []
    if sample_id:
        # 只处理指定的样本
        sample_path = dataset_dir / sample_id
        if sample_path.exists() and sample_path.is_dir():
            sample_dirs.append(sample_path)
        else:
            logging.error(f"Sample {sample_id} not found in {dataset_dir}")
            return
    else:
        # 处理所有样本
        for item in dataset_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                sample_dirs.append(item)
    
    logging.info(f"Found {len(sample_dirs)} samples to process")
    
    # 处理每个样本
    success_count = 0
    fail_count = 0
    
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_id = sample_dir.name
        
        # 查找视频文件（通常是 192x192.mp4）
        video_files = list(sample_dir.glob("*.mp4"))
        if not video_files:
            logging.warning(f"No video file found in {sample_dir}, skipping.")
            fail_count += 1
            continue
        
        # 使用第一个找到的视频文件（通常只有一个）
        video_path = video_files[0]
        
        # 提取特征
        features = process_video(video_path, feature_extractor, batch_size, use_global_features)
        
        if features is not None:
            # 保存特征文件
            output_path = output_dir / f"{sample_id}_features.pt"
            torch.save(features, output_path)
            
            # 同时保存元数据（帧数、特征维度等）
            if use_global_features:
                # 全局特征：[T, 768]
                metadata = {
                    "n_frames": features.shape[0],
                    "feature_dim": features.shape[1],  # 768
                    "feature_type": "global",  # 标记为全局特征
                    "sample_id": sample_id,
                    "video_path": str(video_path),
                }
            else:
                # Patch tokens：[T, 196, 768]
                metadata = {
                    "n_frames": features.shape[0],
                    "n_tokens": features.shape[1],  # 196
                    "feature_dim": features.shape[2],  # 768
                    "feature_type": "patch_tokens",  # 标记为 patch tokens
                    "sample_id": sample_id,
                    "video_path": str(video_path),
                }
            metadata_path = output_dir / f"{sample_id}_metadata.pt"
            torch.save(metadata, metadata_path)
            
            success_count += 1
            logging.debug(f"Saved features for {sample_id}: {features.shape}")
        else:
            fail_count += 1
    
    logging.info(
        f"Processing complete: {success_count} succeeded, {fail_count} failed"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 features from dataset videos for future vision prediction"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset directory containing sample directories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for feature files",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=256,
        help="Embedding dimension (not used for feature extraction, but kept for compatibility)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, or cpu)",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="Process only a specific sample (for testing)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--use_global_features",
        action="store_true",
        default=True,
        help="Extract global features (average over patch tokens) for a⁰ supervision. "
             "If False, extract patch tokens [T, 196, 768]. Default: True",
    )
    parser.add_argument(
        "--use_patch_tokens",
        action="store_true",
        default=False,
        help="Extract patch tokens instead of global features. "
             "This will override --use_global_features if set.",
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # 检查设备
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # 确定使用全局特征还是 patch tokens
    use_global_features = args.use_global_features and not args.use_patch_tokens
    
    if use_global_features:
        logging.info("Using global features (average over patch tokens) for a⁰ supervision")
    else:
        logging.info("Using patch tokens (full 196 tokens per frame)")
    
    # 初始化特征提取器
    logging.info("Initializing DINOv2 feature extractor...")
    feature_extractor = DinoV2FeatureExtractor(device=args.device)
    
    # 处理数据集
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    process_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        sample_id=args.sample_id,
        use_global_features=use_global_features,
    )


if __name__ == "__main__":
    main()
