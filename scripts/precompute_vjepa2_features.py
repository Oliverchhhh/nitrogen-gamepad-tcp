#!/usr/bin/env python3
"""
离线预计算 V-JEPA2.1 视觉表征。

复用 OpenP2P 的视频读取方式（decoder 切片），对每个 chunk 的 video.mp4
按 batch 解码 → GPU resize 到 384×384 → V-JEPA2 推理 → 保存 mean-pooled 表征。

输出: 每个 chunk 目录下 vjepa2_features.pt，[T, embed_dim] float16

用法:
    bash scripts/precompute_vjepa2_features.sh
"""

import argparse
import glob
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elefant.im_tokenizer.tokenizer import Vjepa2Tokenizer
from elefant.im_tokenizer.config import ImageTokenizerConfig, VjepaTokenizerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_tokenizer(args) -> Vjepa2Tokenizer:
    vjepa_config = VjepaTokenizerConfig(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        checkpoint_key="ema_encoder",
        frozen=True,
        img_size=args.img_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        num_frames=64,
        use_hub_fallback=False,
    )
    tokenizer = Vjepa2Tokenizer(
        config=ImageTokenizerConfig(type="vjepa2", vjepa_tokenizer_config=vjepa_config),
        frame_height=args.img_size,
        frame_width=args.img_size,
        embed_dim=args.embed_dim,
    )
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False
    return tokenizer.to(args.device)


def process_chunk(tokenizer, video_path: str, args) -> torch.Tensor:
    """
    复用 OpenP2P 的 decoder 切片方式读取视频，按 batch 推理。
    返回 [T_total, embed_dim] float16。
    """
    decoder = VideoDecoder(video_path, device="cpu", num_ffmpeg_threads=4)
    n_frames = len(decoder)
    all_features = []
    B = args.batch_frames

    for start in range(0, n_frames, B):
        end = min(start + B, n_frames)
        batch_len = end - start

        # 和 OpenP2P 一样用切片读取
        frames = decoder[start:end]  # [T_batch, C, H, W] uint8

        # tubelet_size 补齐
        if frames.shape[0] < args.tubelet_size:
            frames = torch.cat([frames, frames[-1:].expand(
                args.tubelet_size - frames.shape[0], -1, -1, -1)], dim=0)

        # GPU resize + 推理
        x = frames.to(args.device, dtype=torch.float32, non_blocking=True) / 255.0
        if x.shape[-2] != args.img_size or x.shape[-1] != args.img_size:
            x = F.interpolate(x, size=(args.img_size, args.img_size),
                              mode="bilinear", align_corners=False)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = tokenizer(x.unsqueeze(0))          # [1, T, N, D]
            feats = tokens.mean(dim=2)[:, :batch_len]   # [1, batch_len, D]

        all_features.append(feats.squeeze(0).cpu().half())

    return torch.cat(all_features, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video_name", default="video_384.mp4",
                        help="要处理的视频文件名，默认 video_384.mp4（缩放后版本）")
    parser.add_argument("--model_name", default="vjepa2_1_vit_base_384")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--batch_frames", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_name", default="vjepa2_features.pt")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logging.info(f"设备: {args.device}, img_size: {args.img_size}, batch_frames: {args.batch_frames}")

    video_paths = sorted(glob.glob(
        os.path.join(args.data_folder, "**", args.video_name), recursive=True
    ))
    logging.info(f"找到 {len(video_paths)} 个 {args.video_name}")

    if args.num_workers > 1:
        video_paths = video_paths[args.worker_id::args.num_workers]
        logging.info(f"Worker {args.worker_id}/{args.num_workers}: 处理 {len(video_paths)} 个")

    if args.skip_existing:
        before = len(video_paths)
        video_paths = [p for p in video_paths
                       if not os.path.exists(
                           os.path.join(os.path.dirname(p), args.output_name))]
        logging.info(f"跳过已完成 {before - len(video_paths)} 个，剩余 {len(video_paths)} 个")

    if not video_paths:
        logging.info("全部已完成！")
        return

    logging.info("加载 V-JEPA2.1 模型...")
    tokenizer = build_tokenizer(args)
    logging.info("模型加载完成")

    processed = errors = 0
    t0 = time.time()

    for video_path in tqdm(video_paths, desc=f"GPU{args.gpu}"):
        output_path = os.path.join(os.path.dirname(video_path), args.output_name)
        try:
            features = process_chunk(tokenizer, video_path, args)
            torch.save(features, output_path)
            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                logging.info(
                    f"进度 {processed}/{len(video_paths)}, "
                    f"{elapsed/processed:.2f}s/chunk, "
                    f"预计剩余 {(len(video_paths)-processed)*elapsed/processed/60:.1f}min"
                )
        except Exception as e:
            logging.warning(f"失败 {video_path}: {e}")
            errors += 1

    elapsed = time.time() - t0
    logging.info(f"完成: 处理 {processed}, 失败 {errors}, "
                 f"耗时 {elapsed:.1f}s, 平均 {elapsed/max(processed,1):.2f}s/chunk")


if __name__ == "__main__":
    main()
