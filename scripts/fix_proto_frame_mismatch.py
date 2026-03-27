#!/usr/bin/env python3
"""
Fix frame count mismatch between annotation.proto and video files.

For each sample directory under --dataset-root:
1) Read annotation.proto frame_annotations length.
2) Probe decodable frame count from a video file (default: 192x192.mp4).
3) If proto frames > video frames, truncate proto frame_annotations to video frames.
4) Keep a .bak backup by default.

This is intended for already-converted open-p2p style datasets.
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from elefant.data.proto import video_annotation_pb2


def probe_video_frame_count(video_path: Path) -> Optional[int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,avg_frame_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        streams = json.loads(out).get("streams", [])
        if not streams:
            return None
        s = streams[0]
        nb_read_frames = s.get("nb_read_frames")
        if nb_read_frames not in (None, "N/A"):
            return int(nb_read_frames)
        nb_frames = s.get("nb_frames")
        if nb_frames not in (None, "N/A"):
            return int(nb_frames)

        duration = s.get("duration")
        avg_fps = s.get("avg_frame_rate")
        if duration not in (None, "N/A") and avg_fps not in (None, "N/A", "0/0"):
            num, den = avg_fps.split("/")
            fps = float(num) / float(den)
            return int(math.floor(float(duration) * fps + 1e-6))
        return None
    except Exception:
        return None


def pick_video_file(sample_dir: Path, preferred_name: str) -> Optional[Path]:
    preferred = sample_dir / preferred_name
    if preferred.exists():
        return preferred
    fallback = sample_dir / "video.mp4"
    if fallback.exists():
        return fallback
    return None


def load_proto(proto_path: Path) -> video_annotation_pb2.VideoAnnotation:
    data = proto_path.read_bytes()
    va = video_annotation_pb2.VideoAnnotation()
    va.ParseFromString(data)
    return va


def fix_one_sample(
    sample_dir: Path,
    preferred_video_name: str,
    dry_run: bool,
    backup: bool,
) -> Tuple[str, str]:
    proto_path = sample_dir / "annotation.proto"
    if not proto_path.exists():
        return ("skip", "no annotation.proto")

    video_path = pick_video_file(sample_dir, preferred_video_name)
    if video_path is None:
        return ("skip", f"no {preferred_video_name} or video.mp4")

    video_frames = probe_video_frame_count(video_path)
    if video_frames is None:
        return ("skip", f"ffprobe failed for {video_path.name}")

    va = load_proto(proto_path)
    proto_frames = len(va.frame_annotations)

    if proto_frames == video_frames:
        return ("ok", f"aligned ({proto_frames})")

    if proto_frames < video_frames:
        return (
            "warn",
            f"proto shorter than video (proto={proto_frames}, video={video_frames})",
        )

    # proto_frames > video_frames: truncate
    if dry_run:
        return (
            "fix",
            f"would truncate proto {proto_frames} -> {video_frames} ({video_path.name})",
        )

    if backup:
        backup_path = proto_path.with_suffix(".proto.bak")
        if not backup_path.exists():
            shutil.copy2(proto_path, backup_path)

    del va.frame_annotations[video_frames:]
    proto_path.write_bytes(va.SerializeToString())
    return (
        "fix",
        f"truncated proto {proto_frames} -> {video_frames} ({video_path.name})",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fix annotation.proto frame count mismatches against video frame counts."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory of converted dataset (contains sample subdirs).",
    )
    parser.add_argument(
        "--preferred-video",
        type=str,
        default="192x192.mp4",
        help="Preferred video filename for frame probing (default: 192x192.mp4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be changed.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create annotation.proto.bak before modification.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    sample_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not sample_dirs:
        print(f"No sample directories found under {dataset_root}")
        return

    stats = {"ok": 0, "fix": 0, "warn": 0, "skip": 0}
    for sample_dir in sample_dirs:
        status, msg = fix_one_sample(
            sample_dir=sample_dir,
            preferred_video_name=args.preferred_video,
            dry_run=args.dry_run,
            backup=(not args.no_backup),
        )
        stats[status] += 1
        print(f"[{status}] {sample_dir.name}: {msg}")

    print(
        "Summary:",
        f"ok={stats['ok']}",
        f"fix={stats['fix']}",
        f"warn={stats['warn']}",
        f"skip={stats['skip']}",
    )


if __name__ == "__main__":
    main()

