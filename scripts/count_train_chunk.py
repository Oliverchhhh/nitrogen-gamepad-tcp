# scripts/count_train_chunks.py
from pathlib import Path
import logging

import torch
from torchcodec.decoders import VideoDecoder  # 和 video_proto_dataset 一样


def count_chunks_for_video(video_path: Path, T: int) -> int:
    """
    模拟 elefant.data.video_proto_dataset._parse_proto_video_into_chunks
    的切片逻辑（不抓帧，只用 n_frames），返回这一个视频最多能切出的样本数。
    """
    try:
        decoder = VideoDecoder(str(video_path), device="cpu", num_ffmpeg_threads=1)
    except Exception as e:
        logging.warning(f"Failed to open video {video_path}: {e}")
        return 0

    n_frames = len(decoder)
    del decoder

    # 至少要 2 帧，否则直接跳过（和原代码一致）
    if n_frames < 2:
        logging.warning(f"Video {video_path} has less than 2 frames, skipping.")
        return 0

    # 不考虑 shuffle，等价于 start_frame0 = 0
    start_frame = 0
    stride = T
    n_chunks = 0

    # main 部分：每次取 T+1 帧（前 T 帧做输入，最后 1 帧做“look ahead”）
    while start_frame + (T + 1) <= n_frames:
        n_chunks += 1
        start_frame += stride

    # tail 部分：如果还剩至少 2 帧，就 pad 成 1 个 chunk
    if start_frame < n_frames - 1:
        n_chunks += 1

    return n_chunks


def main():
    # 和训练配置一致
    DATASET_ROOT = Path("dataset")
    T = 200  # shared.n_seq_timesteps
    BATCH_SIZE = 4  # config.stage3_finetune.training_dataset.batch_size

    logging.basicConfig(level=logging.INFO)

    all_sample_dirs = sorted(
        d for d in DATASET_ROOT.iterdir() if d.is_dir()
    )
    logging.info(f"Found {len(all_sample_dirs)} sample dirs under {DATASET_ROOT}")

    total_chunks = 0
    per_video_info = []

    for sample_dir in all_sample_dirs:
        # 和训练一样，优先用 192x192.mp4，如果没有就用 video.mp4
        video_192 = sample_dir / "192x192.mp4"
        video_raw = sample_dir / "video.mp4"

        if video_192.exists():
            video_path = video_192
        elif video_raw.exists():
            video_path = video_raw
        else:
            logging.warning(f"No video found in {sample_dir}, skipping.")
            continue

        n_chunks = count_chunks_for_video(video_path, T=T)
        total_chunks += n_chunks
        per_video_info.append((sample_dir.name, n_chunks))

    total_batches = total_chunks / BATCH_SIZE if BATCH_SIZE > 0 else float("nan")

    print("========================================")
    print(f"T = {T}, batch_size = {BATCH_SIZE}")
    print(f"样本数（video chunks）：{total_chunks}")
    print(f"按 batch_size={BATCH_SIZE} 理论上可构成的 batch 数 ≈ {total_batches:.2f}")
    print("========================================")
    print("前几条样本示例：")
    for name, n_chunks in per_video_info[:10]:
        print(f"{name}: {n_chunks} chunks")


if __name__ == "__main__":
    main()