"""
Offline StaMo feature precomputation script.

Walks a dataset directory tree, finds video chunks, and for each chunk
computes StaMo representations for every frame, saving them as
`stamo_features.pt` in the same directory.

Usage:
    python scripts/precompute_stamo_features.py \\
        --dataset_dir cuphead_dataset_converted \\
        --stamo_checkpoint_dir /path/to/StaMo/checkpoint \\
        --video_name 256x256.mp4 \\
        --batch_size 8 \\
        --device cuda

Each stamo_features.pt is a dict:
    {
        "features": Tensor[T, 8192],   # T frames × flattened [2, 4096]
        "frame_count": int,
    }

During training, set use_precomputed_stamo_features: true in the YAML config
and attach `batch.precomputed_stamo_features: [B, T, 8192]` in the data loader.
"""

import argparse
import logging
import os
import sys

import imageio
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_STAMO_REPO = os.environ.get("STAMO_REPO", "/home/ch/StaMo")


def load_stamo_encoder(checkpoint_dir: str) -> "StaMoEncoder":
    if _STAMO_REPO not in sys.path:
        sys.path.insert(0, _STAMO_REPO)
    # Import from the elefant package (adjust path as needed)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from elefant.policy_model.stamo_encoder import StaMoEncoder
    encoder = StaMoEncoder(checkpoint_dir=checkpoint_dir)
    return encoder


def iter_video_chunks(dataset_dir: str, video_name: str):
    """Yield (chunk_dir, video_path) for each chunk containing video_name."""
    for root, dirs, files in os.walk(dataset_dir):
        if video_name in files:
            yield root, os.path.join(root, video_name)


def encode_chunk(
    encoder,
    video_path: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Encode all frames in a video via streaming to avoid OOM. Returns [T, FLAT_DIM] on CPU."""
    all_feats = []
    buf = []
    reader = imageio.get_reader(video_path)
    for frame in reader:
        # frame: [H, W, 3] uint8 numpy array
        buf.append(torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0)
        if len(buf) == batch_size:
            batch = torch.stack(buf).to(device)         # [B, 3, H, W]
            embeds = encoder.encode(batch)              # [B, 2, 4096]
            all_feats.append(embeds.reshape(embeds.shape[0], -1).cpu())
            buf = []
    reader.close()
    if buf:
        batch = torch.stack(buf).to(device)
        embeds = encoder.encode(batch)
        all_feats.append(embeds.reshape(embeds.shape[0], -1).cpu())
    return torch.cat(all_feats, dim=0)  # [T, 8192]


def main():
    parser = argparse.ArgumentParser(description="Precompute StaMo features for training data.")
    parser.add_argument("--dataset_dir", required=True, help="Root directory of the dataset.")
    parser.add_argument("--stamo_checkpoint_dir", required=True, help="StaMo checkpoint directory.")
    parser.add_argument("--video_name", default="256x256.mp4", help="Video filename within each chunk.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for StaMo encoding.")
    parser.add_argument("--device", default="cuda", help="Device for StaMo encoder.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if stamo_features.pt exists.")
    args = parser.parse_args()

    logging.info("Loading StaMo encoder from %s", args.stamo_checkpoint_dir)
    encoder = load_stamo_encoder(args.stamo_checkpoint_dir)
    encoder = encoder.to(args.device)

    chunks = list(iter_video_chunks(args.dataset_dir, args.video_name))
    logging.info("Found %d video chunks in %s", len(chunks), args.dataset_dir)

    for i, (chunk_dir, video_path) in enumerate(chunks):
        out_path = os.path.join(chunk_dir, "stamo_features.pt")
        if os.path.exists(out_path) and not args.overwrite:
            logging.info("[%d/%d] Skipping (already exists): %s", i + 1, len(chunks), chunk_dir)
            continue

        logging.info("[%d/%d] Processing: %s", i + 1, len(chunks), video_path)
        try:
            feats = encode_chunk(encoder, video_path, args.batch_size, args.device)
            torch.save({"features": feats, "frame_count": feats.shape[0]}, out_path)
            logging.info("  Saved %s: shape=%s", out_path, feats.shape)
        except Exception as e:
            logging.error("  Failed to process %s: %s", video_path, e)

    logging.info("Done.")


if __name__ == "__main__":
    main()
