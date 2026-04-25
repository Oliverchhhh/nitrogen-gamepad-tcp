"""
比较 teacher-forcing 和 autoregressive 两个验证日志中的 logit 差异。

用法:
  python scripts/compare_logits.py \
    --tf  val_logs/validate_action_direct_20260423_193612.log \
    --ar  val_logs/validate_ar_action_direct_20260423_193435.log \
    [--max_frames N]   # 只比较前 N 帧，默认全部
    [--plot]           # 生成折线图 logit_diff.png
"""

import re
import argparse
import math
from typing import List, Dict

# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

# 匹配一行中所有浮点数（含负号）
_FLOAT_RE = re.compile(r"-?\d+\.\d+")

# AR 日志：每帧一行，形如
#   GamepadDirectActionLogits(buttons=tensor([[[...]]]), left_stick_x=..., ...)
# TF 日志：每 batch 一行（含多帧），形如
#   [batch N] logits: GamepadDirectActionLogits(buttons=tensor([[[...], [...], ...]]), ...)
_LOGIT_LINE_RE = re.compile(r"GamepadDirectActionLogits\(")


def _extract_field_tensors(line: str) -> Dict[str, List[float]]:
    """
    从一行 GamepadDirectActionLogits(...) 文本中提取各字段的数值列表。
    返回 {'buttons': [...], 'left_stick_x': [...], ...}
    """
    fields = ["buttons", "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"]
    result = {}
    for i, field in enumerate(fields):
        # 找到 field= 之后、下一个 field= 之前的子串
        start = line.find(f"{field}=")
        if start == -1:
            result[field] = []
            continue
        if i + 1 < len(fields):
            end = line.find(f"{fields[i+1]}=", start)
            segment = line[start:end] if end != -1 else line[start:]
        else:
            segment = line[start:]
        result[field] = [float(x) for x in _FLOAT_RE.findall(segment)]
    return result


def parse_log(path: str) -> List[Dict[str, List[float]]]:
    """
    解析日志，返回逐帧的 logit 列表。
    TF 日志每行含多帧（batch 维度展开），AR 日志每行一帧。
    统一展开为 per-frame 列表。
    """
    frames = []
    fields = ["buttons", "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"]
    # 字段维度（buttons=12, sticks=3 each）
    field_dims = {"buttons": 12, "left_stick_x": 3, "left_stick_y": 3,
                  "right_stick_x": 3, "right_stick_y": 3}

    # 多行拼接：有时一个 logit 块跨多行
    buffer = ""
    in_block = False

    with open(path, "r", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if _LOGIT_LINE_RE.search(line):
                in_block = True
                buffer = line
            elif in_block:
                buffer += " " + line.strip()

            if in_block:
                # 判断括号是否闭合
                if buffer.count("(") <= buffer.count(")"):
                    in_block = False
                    field_data = _extract_field_tensors(buffer)
                    # 按帧拆分：buttons 有 12 个值/帧，sticks 有 3 个值/帧
                    n_frames = len(field_data.get("buttons", [])) // field_dims["buttons"]
                    if n_frames == 0:
                        buffer = ""
                        continue
                    for t in range(n_frames):
                        frame = {}
                        for fld in fields:
                            dim = field_dims[fld]
                            vals = field_data.get(fld, [])
                            frame[fld] = vals[t * dim: (t + 1) * dim]
                        frames.append(frame)
                    buffer = ""

    return frames


# ---------------------------------------------------------------------------
# 比较
# ---------------------------------------------------------------------------

def compare(tf_frames, ar_frames, max_frames=None):
    n = min(len(tf_frames), len(ar_frames))
    if max_frames:
        n = min(n, max_frames)

    fields = ["buttons", "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"]

    # 逐帧逐字段 MAE
    per_field_mae: Dict[str, List[float]] = {f: [] for f in fields}
    per_field_max: Dict[str, List[float]] = {f: [] for f in fields}

    for t in range(n):
        tf_f = tf_frames[t]
        ar_f = ar_frames[t]
        for fld in fields:
            tf_v = tf_f.get(fld, [])
            ar_v = ar_f.get(fld, [])
            if not tf_v or not ar_v or len(tf_v) != len(ar_v):
                continue
            diffs = [abs(a - b) for a, b in zip(tf_v, ar_v)]
            per_field_mae[fld].append(sum(diffs) / len(diffs))
            per_field_max[fld].append(max(diffs))

    print(f"\n{'='*60}")
    print(f"比较帧数: {n}  (TF={len(tf_frames)}, AR={len(ar_frames)})")
    print(f"{'='*60}")
    print(f"{'字段':<20} {'平均MAE':>10} {'最大MAE':>10} {'帧级最大差':>12}")
    print(f"{'-'*60}")
    for fld in fields:
        maes = per_field_mae[fld]
        maxs = per_field_max[fld]
        if not maes:
            print(f"{fld:<20} {'N/A':>10}")
            continue
        avg_mae = sum(maes) / len(maes)
        avg_max = sum(maxs) / len(maxs)
        frame_max = max(maxs)
        print(f"{fld:<20} {avg_mae:>10.4f} {avg_max:>10.4f} {frame_max:>12.4f}")
    print(f"{'='*60}\n")

    # 找出差异最大的帧
    total_diff_per_frame = []
    for t in range(n):
        tf_f = tf_frames[t]
        ar_f = ar_frames[t]
        total = 0.0
        cnt = 0
        for fld in fields:
            tf_v = tf_f.get(fld, [])
            ar_v = ar_f.get(fld, [])
            if len(tf_v) == len(ar_v):
                total += sum(abs(a - b) for a, b in zip(tf_v, ar_v))
                cnt += len(tf_v)
        total_diff_per_frame.append(total / cnt if cnt else 0.0)

    top_k = sorted(range(n), key=lambda i: total_diff_per_frame[i], reverse=True)[:5]
    print("差异最大的 5 帧:")
    print(f"{'帧索引':>8} {'平均绝对差':>12}")
    for idx in top_k:
        print(f"{idx:>8} {total_diff_per_frame[idx]:>12.4f}")
    print()

    return per_field_mae, per_field_max, total_diff_per_frame


# ---------------------------------------------------------------------------
# 可选绘图
# ---------------------------------------------------------------------------

def plot(per_field_mae, total_diff, output="logit_diff.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    for fld, maes in per_field_mae.items():
        ax.plot(maes, label=fld, alpha=0.8)
    ax.set_title("逐帧各字段 MAE (TF vs AR)")
    ax.set_xlabel("帧索引")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(total_diff, color="red", alpha=0.8)
    ax2.set_title("逐帧总体平均绝对差")
    ax2.set_xlabel("帧索引")
    ax2.set_ylabel("平均绝对差")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    print(f"图表已保存: {output}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", required=True, help="teacher-forcing 日志路径")
    parser.add_argument("--ar", required=True, help="autoregressive 日志路径")
    parser.add_argument("--max_frames", type=int, default=None, help="最多比较前 N 帧")
    parser.add_argument("--plot", action="store_true", help="生成折线图")
    parser.add_argument("--plot_output", default="logit_diff.png")
    args = parser.parse_args()

    print(f"解析 TF 日志: {args.tf}")
    tf_frames = parse_log(args.tf)
    print(f"  → 解析到 {len(tf_frames)} 帧")

    print(f"解析 AR 日志: {args.ar}")
    ar_frames = parse_log(args.ar)
    print(f"  → 解析到 {len(ar_frames)} 帧")

    per_field_mae, _, total_diff = compare(tf_frames, ar_frames, args.max_frames)

    if args.plot:
        plot(per_field_mae, total_diff, args.plot_output)


if __name__ == "__main__":
    main()
