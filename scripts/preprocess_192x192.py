#!/usr/bin/env python3
"""
批量将 video.mp4 预处理为 192x192.mp4，消除训练时的解码+resize瓶颈。

对已有 bbox 裁剪的 video.mp4（各种分辨率）统一 resize 到 192x192。
对符号链接的 video.mp4（原始分辨率）也统一 resize 到 192x192。

用法:
    python scripts/preprocess_192x192.py \
        --dataset-dir 20260330_cuphead_RJT/cuphead_dataset_converted \
        --jobs 16
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def convert_one(video_path: Path, output_path: Path, crf: int) -> tuple:
    """将单个视频 resize 到 192x192。"""
    if output_path.exists():
        return (str(video_path), True, "skip(exists)")

    # 解析符号链接
    real_input = video_path.resolve() if video_path.is_symlink() else video_path
    if not real_input.exists():
        return (str(video_path), False, "source not found")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(real_input),
        "-vf", "scale=192:192",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(crf),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return (str(video_path), False, result.stderr[-200:])
    return (str(video_path), True, "ok")


def main():
    parser = argparse.ArgumentParser(description="批量预处理视频到 192x192")
    parser.add_argument("--dataset-dir", type=str, required=True, help="数据集根目录")
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF（默认 18）")
    parser.add_argument("--jobs", "-j", type=int, default=8, help="并行进程数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 192x192.mp4")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"错误：目录不存在: {dataset_dir}")
        sys.exit(1)

    # 扫描所有 video.mp4
    video_files = sorted(dataset_dir.rglob("video.mp4"))
    print(f"找到 {len(video_files)} 个 video.mp4")

    if not video_files:
        sys.exit(0)

    tasks = []
    for vf in video_files:
        out = vf.parent / "192x192.mp4"
        if args.overwrite or not out.exists():
            tasks.append((vf, out))

    print(f"需要处理: {len(tasks)} 个（已存在跳过: {len(video_files) - len(tasks)} 个）")

    if not tasks:
        print("全部已完成。")
        sys.exit(0)

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        for vf, out in tasks:
            fut = executor.submit(convert_one, vf, out, args.crf)
            futures[fut] = vf

        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            path, ok, msg = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
                print(f"  失败: {path}: {msg}")

            if i % 100 == 0 or i == total:
                print(f"  进度: {i}/{total}  成功={success}  失败={failed}")

    print(f"\n完成！成功={success}  失败={failed}")


if __name__ == "__main__":
    main()
