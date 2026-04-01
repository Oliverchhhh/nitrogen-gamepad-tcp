#!/usr/bin/env python3
"""
将已转换的 proto 标注和 NitroGen 原始视频组装成 open-p2p 训练期望的目录结构。

核心设计：
  1. 从 proto 的 video_filter_info 字段读取 bbox 信息（无需外部配置文件）
  2. 用 bbox_game_area 裁剪游戏画面 → video.mp4（保持裁剪后原始分辨率）
  3. 用 bbox_controller_overlay 裁剪手柄画面 → gamepad.mp4
  4. 如果没有 bbox_game_area，直接符号链接原始视频
  5. resize 交给训练时的数据预处理（load_video_name="video.mp4"）

bbox 格式（存储在 proto metadata.video_source_info.video_filter_info JSON 中）：
  - bbox_game_area:          {"xtl": 0.0, "ytl": 0.1, "xbr": 0.8, "ybr": 0.9}  归一化 [0,1]
  - bbox_controller_overlay: [x, y, w, h]  绝对像素坐标

输出结构：
  <output-dir>/<video_id>/<chunk_name>/
      ├── annotation.proto
      ├── video.mp4             (裁剪后的游戏画面)
      └── gamepad.mp4           (裁剪后的手柄画面，可选)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

# 延迟导入 proto，避免在 --help 时触发 torch 依赖
_video_annotation_pb2 = None


def _get_pb2():
    global _video_annotation_pb2
    if _video_annotation_pb2 is None:
        from elefant.data.proto import video_annotation_pb2
        _video_annotation_pb2 = video_annotation_pb2
    return _video_annotation_pb2


def parse_video_filter_info(proto_path: str) -> dict:
    """从 proto 文件中解析 video_filter_info JSON。"""
    pb2 = _get_pb2()
    with open(proto_path, "rb") as f:
        va = pb2.VideoAnnotation()
        va.ParseFromString(f.read())
    vfi = va.metadata.video_source_info.video_filter_info
    if not vfi:
        return {}
    return json.loads(vfi)


def find_matching_pairs(annotation_dir: Path, video_dir: Path):
    """
    扫描标注目录和视频目录，找到所有匹配的 (proto, mp4) 对。

    Returns:
        list of (video_id, chunk_name, proto_path, video_path)
    """
    pairs = []
    for vid_dir in sorted(annotation_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        video_id = vid_dir.name
        video_subdir = video_dir / video_id
        if not video_subdir.is_dir():
            continue

        for proto_file in sorted(vid_dir.glob("*.proto")):
            chunk_name = proto_file.stem
            mp4_file = video_subdir / f"{chunk_name}.mp4"
            if mp4_file.exists():
                pairs.append((video_id, chunk_name, proto_file, mp4_file))

    return pairs


def probe_video_nb_frames(video_path: Path) -> Optional[int]:
    """用 ffprobe 获取视频帧数。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return int(out)
    except Exception:
        return None


def align_proto_to_video(proto_path: Path, output_path: Path, video_nb_frames: int) -> Tuple[bool, str]:
    """
    对齐 proto 标注帧数与视频帧数。

    - proto 帧数 > 视频帧数：截断 proto 的 frame_annotations
    - proto 帧数 < 视频帧数：不处理（视频多余帧会被 open-p2p 忽略）
    - proto 帧数 == 视频帧数：直接拷贝

    Returns:
        (changed, message)
    """
    pb2 = _get_pb2()
    with open(proto_path, "rb") as f:
        va = pb2.VideoAnnotation()
        va.ParseFromString(f.read())

    proto_frames = len(va.frame_annotations)

    if proto_frames == video_nb_frames:
        # 完全对齐，直接拷贝
        shutil.copy2(proto_path, output_path)
        return (False, "aligned")

    if proto_frames > video_nb_frames:
        # proto 多于视频，截断标注
        del va.frame_annotations[video_nb_frames:]
        with open(output_path, "wb") as f:
            f.write(va.SerializeToString())
        return (True, f"truncated proto {proto_frames}->{video_nb_frames}")

    # proto 少于视频（视频多余帧被忽略，不需要改 proto）
    shutil.copy2(proto_path, output_path)
    return (False, f"video has extra frames ({video_nb_frames}>{proto_frames}), ok")


def crop_video_normalized_bbox(
    source_path: Path,
    output_path: Path,
    bbox: dict,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "fast",
) -> Tuple[bool, str]:
    """
    用归一化 bbox 裁剪视频（bbox_game_area 格式）。

    bbox: {"xtl": float, "ytl": float, "xbr": float, "ybr": float}
    """
    xtl = float(bbox["xtl"])
    ytl = float(bbox["ytl"])
    xbr = float(bbox["xbr"])
    ybr = float(bbox["ybr"])
    crop_w = xbr - xtl
    crop_h = ybr - ytl
    if crop_w <= 0 or crop_h <= 0:
        return (False, f"invalid normalized bbox: {bbox}")

    vf = f"crop=iw*{crop_w}:ih*{crop_h}:iw*{xtl}:ih*{ytl}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_path),
        "-vf", vf,
        "-an",
        "-c:v", codec, "-preset", preset, "-crf", str(crf),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return (False, result.stderr[-300:])
    return (True, "ok")


def crop_video_pixel_bbox(
    source_path: Path,
    output_path: Path,
    bbox_xywh: list,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "fast",
) -> Tuple[bool, str]:
    """
    用绝对像素 bbox 裁剪视频（bbox_controller_overlay 格式）。

    bbox_xywh: [x, y, w, h] 绝对像素坐标
    """
    x, y, w, h = int(bbox_xywh[0]), int(bbox_xywh[1]), int(bbox_xywh[2]), int(bbox_xywh[3])
    if w <= 0 or h <= 0:
        return (False, f"invalid pixel bbox: {bbox_xywh}")

    vf = f"crop={w}:{h}:{x}:{y}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_path),
        "-vf", vf,
        "-an",
        "-c:v", codec, "-preset", preset, "-crf", str(crf),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return (False, result.stderr[-300:])
    return (True, "ok")


def process_chunk(
    video_id: str,
    chunk_name: str,
    proto_path: Path,
    video_path: Path,
    output_dir: Path,
    filter_info_cache: dict,
    overwrite: bool,
    codec: str,
    crf: int,
    extract_gamepad: bool,
):
    """处理单个 chunk。"""
    chunk_dir = output_dir / video_id / chunk_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    align_msg = ""

    # 获取该 video_id 的 filter_info（同一 video_id 共享 bbox）
    info = filter_info_cache.get(video_id, {})

    # 0. 探测视频帧数，对齐 proto
    video_nb_frames = probe_video_nb_frames(video_path)

    # 1. 拷贝/对齐 proto → annotation.proto
    dst_proto = chunk_dir / "annotation.proto"
    if overwrite or not dst_proto.exists():
        if video_nb_frames is not None:
            changed, align_msg = align_proto_to_video(
                proto_path, dst_proto, video_nb_frames,
            )
        else:
            # 无法探测帧数，直接拷贝
            shutil.copy2(proto_path, dst_proto)
            align_msg = "no probe"

    # 2. 裁剪游戏画面 → video.mp4
    dst_video = chunk_dir / "video.mp4"
    if overwrite or not dst_video.exists():
        game_bbox = info.get("bbox_game_area")
        if game_bbox:
            ok, msg = crop_video_normalized_bbox(
                video_path, dst_video, game_bbox, codec=codec, crf=crf,
            )
            if not ok:
                errors.append(f"game crop: {msg}")
        else:
            # 无 bbox，符号链接原始视频
            if dst_video.is_symlink():
                dst_video.unlink()
            dst_video.symlink_to(video_path.resolve())

    # 3. 裁剪手柄画面 → gamepad.mp4
    if extract_gamepad:
        ctrl_bbox = info.get("bbox_controller_overlay")
        if ctrl_bbox:
            dst_gamepad = chunk_dir / "gamepad.mp4"
            if overwrite or not dst_gamepad.exists():
                ok, msg = crop_video_pixel_bbox(
                    video_path, dst_gamepad, ctrl_bbox, codec=codec, crf=crf,
                )
                if not ok:
                    errors.append(f"gamepad crop: {msg}")

    if errors:
        return (chunk_name, False, "; ".join(errors))
    return (chunk_name, True, align_msg)


def build_filter_info_cache(annotation_dir: Path) -> dict:
    """
    预扫描所有 video_id，从每个 video_id 的第一个 proto 中提取 video_filter_info。

    同一 video_id 下所有 chunk 共享相同的 bbox 配置。
    """
    cache = {}
    for vid_dir in sorted(annotation_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        protos = sorted(vid_dir.glob("*.proto"))
        if not protos:
            continue
        try:
            info = parse_video_filter_info(str(protos[0]))
            cache[vid_dir.name] = info
        except Exception as e:
            print(f"  警告: 解析 {protos[0].name} 的 video_filter_info 失败: {e}")
            cache[vid_dir.name] = {}
    return cache


def main():
    parser = argparse.ArgumentParser(
        description="组装 open-p2p 训练数据集：proto + 视频裁剪 → 每 chunk 一个目录"
    )
    parser.add_argument(
        "--annotation-dir", type=str, required=True,
        help="已转换的 proto 标注目录（包含 <video_id>/ 子目录）",
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="NitroGen 原始视频目录（包含 <video_id>/ 子目录）",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--extract-gamepad", action="store_true",
        help="提取手柄画面为 gamepad.mp4（使用 bbox_controller_overlay）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖已存在的文件",
    )
    parser.add_argument(
        "--codec", type=str, default="libx264",
        help="视频编码器（默认 libx264）",
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="视频质量 CRF（默认 18）",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=4,
        help="并行进程数（默认 4）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印匹配和 bbox 信息，不实际处理",
    )
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if not annotation_dir.is_dir():
        print(f"错误：标注目录不存在: {annotation_dir}")
        sys.exit(1)
    if not video_dir.is_dir():
        print(f"错误：视频目录不存在: {video_dir}")
        sys.exit(1)

    # 预扫描 bbox 信息
    print("从 proto 中提取 bbox 信息...")
    filter_cache = build_filter_info_cache(annotation_dir)
    n_game = sum(1 for v in filter_cache.values() if "bbox_game_area" in v)
    n_ctrl = sum(1 for v in filter_cache.values() if "bbox_controller_overlay" in v)
    print(f"  {len(filter_cache)} 个 video_id: "
          f"{n_game} 有 bbox_game_area, {n_ctrl} 有 bbox_controller_overlay")

    # 扫描匹配对
    print("扫描匹配的 proto + 视频对...")
    pairs = find_matching_pairs(annotation_dir, video_dir)
    video_ids = set(p[0] for p in pairs)
    print(f"找到 {len(pairs)} 个匹配的 chunk（来自 {len(video_ids)} 个 video_id）")

    if not pairs:
        print("没有找到匹配的数据，退出。")
        sys.exit(0)

    if args.dry_run:
        print(f"\n[dry-run] 各 video_id 的 bbox 和 chunk 统计：")
        from collections import Counter
        vid_counts = Counter(p[0] for p in pairs)
        for vid in sorted(vid_counts):
            info = filter_cache.get(vid, {})
            game = "crop" if "bbox_game_area" in info else "symlink"
            ctrl = "crop" if "bbox_controller_overlay" in info else "skip"
            print(f"  {vid}: {vid_counts[vid]:4d} chunks  "
                  f"game={game}  gamepad={ctrl}")
        print(f"\n  总计: {len(pairs)} chunks")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 并行处理
    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        for video_id, chunk_name, proto_path, video_path in pairs:
            fut = executor.submit(
                process_chunk,
                video_id, chunk_name, proto_path, video_path,
                output_dir, filter_cache, args.overwrite,
                args.codec, args.crf, args.extract_gamepad,
            )
            futures[fut] = chunk_name

        total = len(futures)
        truncated = 0
        for i, fut in enumerate(as_completed(futures), 1):
            chunk_name, ok, msg = fut.result()
            if ok:
                success += 1
                if "truncated" in msg:
                    truncated += 1
                    if truncated <= 20:
                        print(f"  对齐截断: {chunk_name}: {msg}")
            else:
                failed += 1
                print(f"  失败: {chunk_name}: {msg}")

            if i % 200 == 0 or i == total:
                print(f"  进度: {i}/{total}  成功={success}  失败={failed}  截断={truncated}")

    print(f"\n完成！成功={success}  失败={failed}  proto截断对齐={truncated}")
    print(f"输出目录: {output_dir}")
    print(f"\n训练配置:")
    print(f"  local_prefix: \"{output_dir}\"")
    print(f"  load_video_name: \"video.mp4\"")


if __name__ == "__main__":
    main()
