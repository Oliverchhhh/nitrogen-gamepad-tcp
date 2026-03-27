#!/usr/bin/env python3
"""
Deterministically split an open-p2p style dataset by video key hash.

Default split:
  - train: [0.0, 0.8)
  - val:   [0.8, 0.9)
  - test:  [0.9, 1.0)

Each sample directory must contain both annotation.proto and 192x192.mp4.
The split is done at video level (all chunks of one video stay in the same split).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CHUNK_PATTERN = re.compile(r"^(?P<video_key>.+)_chunk_\d+$")


@dataclass(frozen=True)
class SplitThresholds:
    train_max: float
    val_max: float

    def validate(self) -> None:
        if not (0.0 < self.train_max < 1.0):
            raise ValueError("train_max must be in (0, 1)")
        if not (self.train_max < self.val_max < 1.0):
            raise ValueError("val_max must be in (train_max, 1)")


def hash_to_unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return value / float(1 << 64)


def infer_video_key(sample_dir_name: str) -> str:
    # Typical NitroGen converted name: cuphead_v350335326_chunk_0000
    m = CHUNK_PATTERN.match(sample_dir_name)
    if m:
        return m.group("video_key")
    # Fallback: if no chunk suffix, split by full sample name.
    return sample_dir_name


def discover_samples(dataset_root: Path) -> List[Path]:
    samples: List[Path] = []
    missing_video: List[str] = []
    for entry in sorted(dataset_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "annotation.proto").exists():
            continue
        if not (entry / "192x192.mp4").exists():
            missing_video.append(entry.name)
            continue
        samples.append(entry)
    if missing_video:
        preview = ", ".join(missing_video[:10])
        suffix = " ..." if len(missing_video) > 10 else ""
        raise RuntimeError(
            "Found sample directories with annotation.proto but missing 192x192.mp4: "
            f"{preview}{suffix}. "
            f"Missing count={len(missing_video)}"
        )
    return samples


def choose_split(score: float, thresholds: SplitThresholds) -> str:
    if score < thresholds.train_max:
        return "train"
    if score < thresholds.val_max:
        return "val"
    return "test"


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src, dst, target_is_directory=True)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def summarize(assignments: Iterable[Tuple[str, Path, str, float]]) -> Dict[str, object]:
    counts = {"train": 0, "val": 0, "test": 0}
    videos_per_split: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for video_key, _sample, split, _score in assignments:
        counts[split] += 1
        videos_per_split[split].add(video_key)
    return {
        "sample_counts": counts,
        "video_counts": {k: len(v) for k, v in videos_per_split.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically split dataset by video hash."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input dataset root, e.g. NitroGen_cuphead_3",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=(
            "Output split root. "
            "Will create <output_root>/train, <output_root>/val, <output_root>/test"
        ),
    )
    parser.add_argument(
        "--train_max",
        type=float,
        default=0.8,
        help="Upper bound for train split in [0,1).",
    )
    parser.add_argument(
        "--val_max",
        type=float,
        default=0.9,
        help="Upper bound for val split in [0,1).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["symlink"],
        default="symlink",
        help="How to materialize split directories.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="Optional jsonl manifest path to save per-sample split decisions.",
    )
    args = parser.parse_args()

    thresholds = SplitThresholds(train_max=args.train_max, val_max=args.val_max)
    thresholds.validate()

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    samples = discover_samples(input_dir)
    if not samples:
        raise RuntimeError(
            "No valid sample directories found under "
            f"{input_dir}. Expected both annotation.proto and 192x192.mp4."
        )

    split_dirs = {
        "train": output_root / "train",
        "val": output_root / "val",
        "test": output_root / "test",
    }
    for d in split_dirs.values():
        ensure_clean_dir(d)

    assignments: List[Tuple[str, Path, str, float]] = []
    for sample in samples:
        video_key = infer_video_key(sample.name)
        score = hash_to_unit_interval(video_key)
        split = choose_split(score, thresholds)
        link_or_copy(sample, split_dirs[split] / sample.name, args.mode)
        assignments.append((video_key, sample, split, score))

    summary = summarize(assignments)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.manifest_path:
        manifest_path = Path(args.manifest_path).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            for video_key, sample, split, score in assignments:
                f.write(
                    json.dumps(
                        {
                            "sample": sample.name,
                            "video_key": video_key,
                            "split": split,
                            "score": score,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
