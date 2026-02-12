# pip install -U huggingface_hub tqdm

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "nvidia/NitroGen"
DEFAULT_OUT_DIR = Path("/mnt/d/project/open-p2p-main/nitrogen_datasets")
# DEFAULT_SHARDS = [0, 1]  # 默认下载前 2 个 shard（每个大约 1.3~1.9GB 左右）
DEFAULT_SHARDS = [0]


def main():
    parser = argparse.ArgumentParser(
        description="Download NitroGen dataset subset from Hugging Face"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=DEFAULT_SHARDS,
        help=f"Shard indices to download (default: {DEFAULT_SHARDS})",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help=(
            "Hugging Face Hub endpoint / mirror, e.g. "
            "https://hf-mirror.com; if set, will be written to HF_ENDPOINT"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    shards = args.shards

    # 如果指定了镜像端点，优先使用镜像（通过环境变量生效）
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
        # 一些版本也读取这个环境变量，顺便一起设置
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = args.endpoint
        print(f"Using Hugging Face mirror endpoint: {args.endpoint}")

    # 创建输出目录（如果不存在）
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"Error: Cannot create output directory {out_dir}: {e}")
        sys.exit(1)

    allow_patterns = [f"actions/SHARD_{i:04d}.tar.gz" for i in shards] + ["README.md"]

    print(f"Downloading NitroGen dataset subset (shards {shards})...")
    print(f"Output directory: {out_dir}")
    print(f"Files to download: {allow_patterns}")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,  # Windows/无管理员权限时更稳
        )
        print(f"\n✓ Download completed!")
        print(f"Downloaded to: {out_dir.resolve()}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
