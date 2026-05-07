#!/usr/bin/env python3
"""统计数据集中所有按钮的使用频率。

在 count_gamepad_actions.py 的基础上，补充 face buttons / bumpers / start / select
的逐键计数，并额外统计：
  - 每帧同时按下的按钮数分布
  - 按钮共现矩阵（哪些按钮经常同时按下）
  - 按钮 + 摇杆同时激活的比例

输出 JSON 保存到 dataset_jsons/cuphead_button_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from elefant.data.proto import video_annotation_pb2

# 与 CUPHEAD_BUTTON_VOCAB 保持一致的顺序
BUTTON_VOCAB = [
    "south",
    "north",
    "east",
    "west",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "start",
    "select",
    "left_bumper",
    "right_bumper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计数据集按钮使用频率")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("cuphead_one_level_3"),
    )
    parser.add_argument(
        "--stick_deadzone",
        type=float,
        default=0.15,
        help="摇杆死区，用于判断摇杆是否激活",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("dataset_jsons/cuphead_one_level_3_stats.json"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行读取 proto 文件的进程数",
    )
    return parser.parse_args()


def pick_gamepad(frame_annotation):
    sa = frame_annotation.system_action
    ua = frame_annotation.user_action
    if sa.is_known and sa.HasField("game_pad"):
        return sa.game_pad
    if ua.is_known and ua.HasField("game_pad"):
        return ua.game_pad
    return None


def get_pressed(gp) -> list[str]:
    return [btn for btn in BUTTON_VOCAB if getattr(gp.buttons, btn, False)]


def process_proto(proto_path: Path, stick_deadzone: float) -> dict:
    """处理单个 proto 文件，返回局部统计 dict。"""
    va = video_annotation_pb2.VideoAnnotation()
    va.ParseFromString(proto_path.read_bytes())

    btn_counts = Counter({b: 0 for b in BUTTON_VOCAB})
    simultaneous_dist = Counter()          # 每帧同时按下几个键
    cooccur = defaultdict(int)             # 共现计数 (btn_a, btn_b)
    btn_with_left_stick = Counter()        # 按钮 + 左摇杆同时激活
    btn_with_right_stick = Counter()       # 按钮 + 右摇杆同时激活
    total = 0
    total_with_gamepad = 0

    for fa in va.frame_annotations:
        total += 1
        gp = pick_gamepad(fa)
        if gp is None:
            continue
        total_with_gamepad += 1

        pressed = get_pressed(gp)
        for b in pressed:
            btn_counts[b] += 1

        n = len(pressed)
        simultaneous_dist[n] += 1

        # 共现（只统计 n>=2 的情况）
        for i in range(len(pressed)):
            for j in range(i + 1, len(pressed)):
                key = (pressed[i], pressed[j])
                cooccur[key] += 1

        # 摇杆激活判断
        lx, ly = float(gp.left_stick.x), float(gp.left_stick.y)
        rx, ry = float(gp.right_stick.x), float(gp.right_stick.y)
        left_active = math.sqrt(lx * lx + ly * ly) > stick_deadzone
        right_active = math.sqrt(rx * rx + ry * ry) > stick_deadzone
        for b in pressed:
            if left_active:
                btn_with_left_stick[b] += 1
            if right_active:
                btn_with_right_stick[b] += 1

    return {
        "total": total,
        "total_with_gamepad": total_with_gamepad,
        "btn_counts": dict(btn_counts),
        "simultaneous_dist": dict(simultaneous_dist),
        "cooccur": {f"{a}+{b}": v for (a, b), v in cooccur.items()},
        "btn_with_left_stick": dict(btn_with_left_stick),
        "btn_with_right_stick": dict(btn_with_right_stick),
    }


def merge(results: list[dict]) -> dict:
    total = sum(r["total"] for r in results)
    total_gp = sum(r["total_with_gamepad"] for r in results)

    btn_counts: Counter = Counter()
    sim_dist: Counter = Counter()
    cooccur: Counter = Counter()
    btn_ls: Counter = Counter()
    btn_rs: Counter = Counter()

    for r in results:
        btn_counts.update(r["btn_counts"])
        sim_dist.update({int(k): v for k, v in r["simultaneous_dist"].items()})
        cooccur.update(r["cooccur"])
        btn_ls.update(r["btn_with_left_stick"])
        btn_rs.update(r["btn_with_right_stick"])

    return {
        "total": total,
        "total_with_gamepad": total_gp,
        "btn_counts": btn_counts,
        "sim_dist": sim_dist,
        "cooccur": cooccur,
        "btn_ls": btn_ls,
        "btn_rs": btn_rs,
    }


def build_summary(merged: dict, stick_deadzone: float, dataset_root: str,
                  n_proto: int) -> dict:
    total = merged["total"]
    total_gp = merged["total_with_gamepad"]
    btn_counts = merged["btn_counts"]

    # 按频率排序的 per-button 统计
    per_button = {}
    for btn in BUTTON_VOCAB:
        cnt = btn_counts.get(btn, 0)
        rate = cnt / total_gp if total_gp > 0 else 0.0
        per_button[btn] = {
            "count": cnt,
            "rate": round(rate, 6),
            "rate_pct": round(rate * 100, 4),
            "with_left_stick_count": merged["btn_ls"].get(btn, 0),
            "with_right_stick_count": merged["btn_rs"].get(btn, 0),
        }

    # 同时按键数分布
    sim_dist = {
        str(k): {
            "count": v,
            "rate_pct": round(v / total_gp * 100, 4) if total_gp > 0 else 0.0,
        }
        for k, v in sorted(merged["sim_dist"].items())
    }

    # 共现 top-20
    cooccur_top20 = [
        {"pair": pair, "count": cnt}
        for pair, cnt in merged["cooccur"].most_common(20)
    ]

    return {
        "dataset_root": dataset_root,
        "annotation_proto_files": n_proto,
        "stick_deadzone": stick_deadzone,
        "total_frames": total,
        "total_frames_with_gamepad": total_gp,
        "per_button": per_button,
        "simultaneous_button_count_distribution": sim_dist,
        "cooccurrence_top20": cooccur_top20,
    }


def print_summary(summary: dict) -> None:
    total_gp = summary["total_frames_with_gamepad"]
    print("=" * 70)
    print("CUPHEAD BUTTON STATISTICS")
    print("=" * 70)
    print(f"dataset_root          : {summary['dataset_root']}")
    print(f"annotation_proto_files: {summary['annotation_proto_files']}")
    print(f"total_frames          : {summary['total_frames']}")
    print(f"frames_with_gamepad   : {total_gp}")
    print(f"stick_deadzone        : {summary['stick_deadzone']}")
    print()

    print(f"{'BUTTON':<16} {'COUNT':>10} {'RATE%':>8}  {'W/L-STICK':>10}  {'W/R-STICK':>10}")
    print("-" * 70)
    sorted_btns = sorted(
        summary["per_button"].items(),
        key=lambda kv: kv[1]["count"],
        reverse=True,
    )
    for btn, s in sorted_btns:
        ls_pct = (s["with_left_stick_count"] / s["count"] * 100) if s["count"] > 0 else 0.0
        rs_pct = (s["with_right_stick_count"] / s["count"] * 100) if s["count"] > 0 else 0.0
        print(
            f"  {btn:<14} {s['count']:>10,}  {s['rate_pct']:>7.4f}%"
            f"  {ls_pct:>8.1f}%   {rs_pct:>8.1f}%"
        )

    print()
    print("SIMULTANEOUS BUTTON COUNT DISTRIBUTION:")
    for k, v in summary["simultaneous_button_count_distribution"].items():
        bar = "#" * min(40, int(v["rate_pct"] * 0.5))
        print(f"  {k} buttons: {v['count']:>10,}  ({v['rate_pct']:>6.2f}%)  {bar}")

    print()
    print("TOP-20 BUTTON CO-OCCURRENCES:")
    for item in summary["cooccurrence_top20"]:
        rate_pct = item["count"] / total_gp * 100 if total_gp > 0 else 0.0
        print(f"  {item['pair']:<30} {item['count']:>8,}  ({rate_pct:.4f}%)")

    print("=" * 70)


def main() -> None:
    args = parse_args()
    root = args.dataset_root

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    proto_files = sorted(root.rglob("annotation.proto"))
    if not proto_files:
        raise FileNotFoundError(f"No annotation.proto found under: {root}")

    print(f"Found {len(proto_files)} proto files, processing...")

    if args.workers > 1:
        from multiprocessing import Pool
        from functools import partial
        worker = partial(process_proto, stick_deadzone=args.stick_deadzone)
        with Pool(args.workers) as pool:
            results = pool.map(worker, proto_files)
    else:
        results = [process_proto(p, args.stick_deadzone) for p in proto_files]

    merged = merge(results)
    summary = build_summary(merged, args.stick_deadzone, str(root), len(proto_files))

    print_summary(summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
