#!/usr/bin/env python3
"""Count gamepad action usage in a converted dataset.

This script recursively scans `annotation.proto` files under a dataset root and
reports action counts for:
1) D-pad arrow buttons.
2) Left/right stick direction buckets (8-way + neutral).
3) Left/right stick magnitude buckets.

By default, action source selection follows training-time logic:
- Prefer `system_action.game_pad` if available.
- Otherwise use `user_action.game_pad`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from elefant.data.proto import video_annotation_pb2


DPAD_BUTTONS = ("dpad_up", "dpad_down", "dpad_left", "dpad_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count D-pad and stick usage from annotation.proto files."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("cuphead_dataset_converted"),
        help="Dataset root directory to scan recursively.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="Stick deadzone used for neutral classification.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to save all stats as JSON.",
    )
    return parser.parse_args()


def pick_gamepad_action(frame_annotation) -> tuple[Any | None, str]:
    """Pick gamepad action using the same preference as dataset parser."""
    system_action = frame_annotation.system_action
    user_action = frame_annotation.user_action

    if system_action.is_known and system_action.HasField("game_pad"):
        return system_action.game_pad, "system"
    if user_action.is_known and user_action.HasField("game_pad"):
        return user_action.game_pad, "user"
    return None, "none"


def get_stick_direction(x: float, y: float, deadzone: float) -> str:
    """Map stick vector to one of 8 directions or neutral."""
    mag = math.sqrt(x * x + y * y)
    if mag <= deadzone:
        return "neutral"

    angle = math.degrees(math.atan2(y, x))
    if -22.5 <= angle < 22.5:
        return "right"
    if 22.5 <= angle < 67.5:
        return "up_right"
    if 67.5 <= angle < 112.5:
        return "up"
    if 112.5 <= angle < 157.5:
        return "up_left"
    if angle >= 157.5 or angle < -157.5:
        return "left"
    if -157.5 <= angle < -112.5:
        return "down_left"
    if -112.5 <= angle < -67.5:
        return "down"
    return "down_right"


def get_magnitude_bucket(magnitude: float, deadzone: float) -> str:
    """Bucket stick magnitude for coarse strength statistics."""
    if magnitude <= deadzone:
        return "neutral"
    if magnitude <= 0.25:
        return "(0,0.25]"
    if magnitude <= 0.50:
        return "(0.25,0.50]"
    if magnitude <= 0.75:
        return "(0.50,0.75]"
    if magnitude <= 1.00:
        return "(0.75,1.00]"
    return ">1.00"


def update_axis_counts(counter: Counter[str], x: float, y: float, deadzone: float) -> None:
    """Track per-axis directional counts with deadzone."""
    if x > deadzone:
        counter["x_right"] += 1
    elif x < -deadzone:
        counter["x_left"] += 1
    else:
        counter["x_neutral"] += 1

    if y > deadzone:
        counter["y_up"] += 1
    elif y < -deadzone:
        counter["y_down"] += 1
    else:
        counter["y_neutral"] += 1


def counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: kv[0]))


def main() -> None:
    args = parse_args()
    root = args.dataset_root

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    proto_files = sorted(root.rglob("annotation.proto"))
    if not proto_files:
        raise FileNotFoundError(f"No annotation.proto found under: {root}")

    dpad_counts = Counter({k: 0 for k in DPAD_BUTTONS})
    left_dir_counts = Counter()
    right_dir_counts = Counter()
    left_mag_counts = Counter()
    right_mag_counts = Counter()
    left_axis_counts = Counter()
    right_axis_counts = Counter()
    selected_source_counts = Counter({"system": 0, "user": 0, "none": 0})

    total_frames = 0
    total_frames_with_selected_gamepad = 0

    for proto_path in proto_files:
        va = video_annotation_pb2.VideoAnnotation()
        va.ParseFromString(proto_path.read_bytes())

        for frame_annotation in va.frame_annotations:
            total_frames += 1
            gamepad_action, source = pick_gamepad_action(frame_annotation)
            selected_source_counts[source] += 1

            if gamepad_action is None:
                continue

            total_frames_with_selected_gamepad += 1

            for btn in DPAD_BUTTONS:
                if getattr(gamepad_action.buttons, btn):
                    dpad_counts[btn] += 1

            lx = float(gamepad_action.left_stick.x)
            ly = float(gamepad_action.left_stick.y)
            rx = float(gamepad_action.right_stick.x)
            ry = float(gamepad_action.right_stick.y)

            left_dir_counts[get_stick_direction(lx, ly, args.deadzone)] += 1
            right_dir_counts[get_stick_direction(rx, ry, args.deadzone)] += 1

            left_mag = math.sqrt(lx * lx + ly * ly)
            right_mag = math.sqrt(rx * rx + ry * ry)
            left_mag_counts[get_magnitude_bucket(left_mag, args.deadzone)] += 1
            right_mag_counts[get_magnitude_bucket(right_mag, args.deadzone)] += 1

            update_axis_counts(left_axis_counts, lx, ly, args.deadzone)
            update_axis_counts(right_axis_counts, rx, ry, args.deadzone)

    def rate(v: int, denom: int) -> float:
        if denom <= 0:
            return 0.0
        return float(v) / float(denom)

    summary = {
        "dataset_root": str(root),
        "annotation_proto_files": len(proto_files),
        "deadzone": args.deadzone,
        "total_frames": total_frames,
        "frames_with_selected_gamepad": total_frames_with_selected_gamepad,
        "selected_source_counts": counter_to_sorted_dict(selected_source_counts),
        "dpad_counts": counter_to_sorted_dict(dpad_counts),
        "dpad_rates_over_selected_gamepad_frames": {
            k: round(rate(v, total_frames_with_selected_gamepad), 6)
            for k, v in counter_to_sorted_dict(dpad_counts).items()
        },
        "left_stick_direction_counts": counter_to_sorted_dict(left_dir_counts),
        "right_stick_direction_counts": counter_to_sorted_dict(right_dir_counts),
        "left_stick_magnitude_buckets": counter_to_sorted_dict(left_mag_counts),
        "right_stick_magnitude_buckets": counter_to_sorted_dict(right_mag_counts),
        "left_stick_axis_counts": counter_to_sorted_dict(left_axis_counts),
        "right_stick_axis_counts": counter_to_sorted_dict(right_axis_counts),
    }

    print("=" * 80)
    print("GAMEPAD ACTION COUNTS")
    print("=" * 80)
    print(f"dataset_root: {summary['dataset_root']}")
    print(f"annotation_proto_files: {summary['annotation_proto_files']}")
    print(f"deadzone: {summary['deadzone']}")
    print(f"total_frames: {summary['total_frames']}")
    print(
        f"frames_with_selected_gamepad: {summary['frames_with_selected_gamepad']}"
    )
    print(f"selected_source_counts: {summary['selected_source_counts']}")
    print()
    print("DPAD COUNTS:")
    for k, v in summary["dpad_counts"].items():
        r = summary["dpad_rates_over_selected_gamepad_frames"][k]
        print(f"  {k:>10s}: {v:8d} ({r:.4%})")
    print()
    print("LEFT STICK DIRECTION COUNTS:")
    for k, v in summary["left_stick_direction_counts"].items():
        print(f"  {k:>10s}: {v:8d}")
    print()
    print("RIGHT STICK DIRECTION COUNTS:")
    for k, v in summary["right_stick_direction_counts"].items():
        print(f"  {k:>10s}: {v:8d}")
    print()
    print("LEFT STICK MAGNITUDE BUCKETS:")
    for k, v in summary["left_stick_magnitude_buckets"].items():
        print(f"  {k:>10s}: {v:8d}")
    print()
    print("RIGHT STICK MAGNITUDE BUCKETS:")
    for k, v in summary["right_stick_magnitude_buckets"].items():
        print(f"  {k:>10s}: {v:8d}")
    print()
    print("LEFT STICK AXIS COUNTS:")
    for k, v in summary["left_stick_axis_counts"].items():
        print(f"  {k:>10s}: {v:8d}")
    print()
    print("RIGHT STICK AXIS COUNTS:")
    for k, v in summary["right_stick_axis_counts"].items():
        print(f"  {k:>10s}: {v:8d}")
    print("=" * 80)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
