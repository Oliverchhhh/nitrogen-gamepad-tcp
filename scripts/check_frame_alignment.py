#!/usr/bin/env python3
"""
检查不满 1200 帧的视频与 proto 标注的帧数对齐情况。

对比：
  - 视频实际帧数（ffprobe nb_frames）
  - proto 中 frame_annotations 的数量
  - 是否对齐
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# 需要在项目虚拟环境中运行
from elefant.data.proto import video_annotation_pb2


ANNOTATION_DIR = Path("/media/user/LLY/20260330_cuphead_RJT/cuphead_dataset_annotation/cuphead")
VIDEO_DIR = Path("/media/user/LLY/20260330_cuphead_RJT/NitroGen_cuphead_oss")

# 从 /tmp/frame_counts.txt 中筛选出的异常视频
ABNORMAL_VIDEOS = """
600 ./v2231979338/v2231979338_chunk_0001.mp4
720 ./v1866813598/v1866813598_chunk_0054.mp4
900 ./v1227971819/v1227971819_chunk_0155.mp4
1020 ./v1491189952/v1491189952_chunk_0111.mp4
1199 ./v2032107522/v2032107522_chunk_0000.mp4
1199 ./v2569165655/v2569165655_chunk_0000.mp4
1201 ./v1120198605/v1120198605_chunk_0000.mp4
1201 ./v1181221201/v1181221201_chunk_0000.mp4
1201 ./v1211190238/v1211190238_chunk_0000.mp4
1201 ./v1437603604/v1437603604_chunk_0000.mp4
1201 ./v1805597575/v1805597575_chunk_0000.mp4
1201 ./v1838930367/v1838930367_chunk_0000.mp4
1201 ./v1866813598/v1866813598_chunk_0000.mp4
1201 ./v1997087454/v1997087454_chunk_0000.mp4
1201 ./v2010536499/v2010536499_chunk_0000.mp4
1201 ./v2090821160/v2090821160_chunk_0000.mp4
1202 ./v1172215160/v1172215160_chunk_0000.mp4
1202 ./v1491189952/v1491189952_chunk_0000.mp4
1202 ./v1557152215/v1557152215_chunk_0000.mp4
1202 ./v1786888064/v1786888064_chunk_0000.mp4
1202 ./v1873564059/v1873564059_chunk_0000.mp4
1203 ./v1227971819/v1227971819_chunk_0000.mp4
1203 ./v1642130426/v1642130426_chunk_0000.mp4
1203 ./v1895689053/v1895689053_chunk_0000.mp4
""".strip()


def probe_nb_frames(video_path: Path) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "csv=p=0",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out)


def get_proto_frame_count(video_id: str, chunk_name: str) -> int:
    proto_path = ANNOTATION_DIR / video_id / f"{chunk_name}.proto"
    if not proto_path.exists():
        return -1
    with open(proto_path, "rb") as f:
        va = video_annotation_pb2.VideoAnnotation()
        va.ParseFromString(f.read())
    return len(va.frame_annotations)


def main():
    # 解析异常视频列表
    entries = []
    for line in ABNORMAL_VIDEOS.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        nb_frames = int(parts[0])
        rel_path = parts[1]  # ./v2231979338/v2231979338_chunk_0001.mp4
        # 提取 video_id 和 chunk_name
        segments = rel_path.replace("./", "").split("/")
        video_id = segments[0]
        chunk_name = segments[1].replace(".mp4", "")
        entries.append((video_id, chunk_name, nb_frames))

    # 对 v2231979338 只取代表性样本（全部都是 600 帧）
    v2231_seen = False
    filtered = []
    for video_id, chunk_name, nb_frames in entries:
        if video_id == "v2231979338":
            if not v2231_seen:
                v2231_seen = True
                filtered.append((video_id, chunk_name, nb_frames))
            # 跳过其余
        else:
            filtered.append((video_id, chunk_name, nb_frames))

    print(f"{'video_id':<16} {'chunk_name':<35} {'video_frames':>12} {'proto_frames':>12} {'diff':>6} {'status'}")
    print("-" * 105)

    mismatch_count = 0
    for video_id, chunk_name, nb_frames in filtered:
        proto_frames = get_proto_frame_count(video_id, chunk_name)
        if proto_frames == -1:
            status = "NO_PROTO"
            diff = "N/A"
        else:
            d = nb_frames - proto_frames
            diff = f"{d:+d}"
            if d == 0:
                status = "OK"
            else:
                status = "MISMATCH"
                mismatch_count += 1

        print(f"{video_id:<16} {chunk_name:<35} {nb_frames:>12} {proto_frames:>12} {diff:>6} {status}")

    # v2231979338 汇总
    v2231_chunks = [(vid, cn, nf) for vid, cn, nf in entries if vid == "v2231979338"]
    if v2231_chunks:
        print(f"\n--- v2231979338 汇总（全部 600 帧，共 {len(v2231_chunks)} 个 chunk）---")
        # 抽查几个的 proto 帧数
        sample_indices = [0, len(v2231_chunks)//2, -1]
        for idx in sample_indices:
            vid, cn, nf = v2231_chunks[idx]
            pf = get_proto_frame_count(vid, cn)
            d = nf - pf if pf != -1 else "N/A"
            print(f"  {cn}: video={nf}, proto={pf}, diff={d}")

    print(f"\n总计检查: {len(filtered)} 个（v2231979338 取 1 个代表）")
    print(f"不对齐: {mismatch_count} 个")


if __name__ == "__main__":
    main()
