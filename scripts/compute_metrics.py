#!/usr/bin/env python3
"""
compute_metrics.py

读取 NitroGen 模型的 eval.jsonl 推理结果，将连续摇杆/扳机量化为
openp2p gamepad 离散 bin，然后计算与 GT 的对齐指标。

支持两种按钮评估模式（--mode）：
  onehot  （默认）：NitroGen 原始 16 维独立二分类，每维独立计算 accuracy/F1
  slot            ：将 16 维 one-hot 转换为 openp2p slot 编码（6个slot，每个slot是按钮ID），
                    exact_match / accuracy / F1 与 openp2p AR 验证口径完全一致

支持两种 eval.jsonl 格式：
  旧格式：每条记录含 "gt"（单帧），pred[action_shift] 对应该 GT。
  新格式：每条记录含 "gt_sequence"（action_horizon 帧列表），
          pred[k] ↔ gt_sequence[k]，跳过前 action_shift 帧。

NitroGen 按钮 21 维（字母序）→ openp2p slot ID 映射：
  BACK(0)→select(10)  DPAD_DOWN(1)→dpad_down(6)  DPAD_LEFT(2)→dpad_left(7)
  DPAD_RIGHT(3)→dpad_right(8)  DPAD_UP(4)→dpad_up(5)  EAST(5)→east(3)
  LEFT_SHOULDER(7)→left_bumper(11)  LEFT_THUMB(8)→left_thumb(13)
  LEFT_TRIGGER(9)→（扳机按钮位，不映射到slot）
  NORTH(10)→north(2)  RIGHT_SHOULDER(14)→right_bumper(12)
  RIGHT_THUMB(15)→right_thumb(14)  RIGHT_TRIGGER(16)→（扳机按钮位，不映射到slot）
  SOUTH(18)→south(1)  START(19)→start(9)  WEST(20)→west(4)

用法:
    python scripts/compute_metrics.py --mode onehot   # 默认，16维独立二分类
    python scripts/compute_metrics.py --mode slot     # 转换为openp2p slot编码
"""

import argparse
import json
import math
from pathlib import Path

# ── openp2p gamepad 量化参数（与 action_mapping.py 保持一致）──────────────
N_STICK_BINS = 33
N_TRIGGER_BINS = 17
STICK_DEADZONE = 0.05
STICK_CENTER = N_STICK_BINS // 2   # = 16
STICK_DZ_HALF = 1                  # center ± 1 视为死区 bin

# ── NitroGen 21 维按钮名称（字母序）──────────────────────────────────────
BUTTON_ACTION_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE',
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_BOTTOM', 'RIGHT_LEFT',
    'RIGHT_RIGHT', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_TRIGGER', 'RIGHT_UP', 'SOUTH',
    'START', 'WEST'
]
N_BUTTONS = len(BUTTON_ACTION_TOKENS)  # 21

# ── openp2p 匹配按钮：排除无 proto 对应字段的无效按钮 ─────────────────────
# 无效：GUIDE(6) RIGHT_BOTTOM(11) RIGHT_LEFT(12) RIGHT_RIGHT(13) RIGHT_UP(17)
VALID_BUTTON_INDICES = [i for i, name in enumerate(BUTTON_ACTION_TOKENS)
                        if name not in {'GUIDE', 'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_UP'}]
VALID_BUTTON_NAMES = [BUTTON_ACTION_TOKENS[i] for i in VALID_BUTTON_INDICES]
N_VALID_BUTTONS = len(VALID_BUTTON_INDICES)  # 16

# ── openp2p slot 编码参数 ─────────────────────────────────────────────────
# NitroGen 按钮名（大写）→ openp2p slot ID
# LEFT_TRIGGER / RIGHT_TRIGGER 是扳机按钮位，openp2p 里扳机是连续量，不进 slot
_NITROGEN_TO_OPENP2P_ID = {
    'BACK':           10,  # select
    'DPAD_DOWN':       6,
    'DPAD_LEFT':       7,
    'DPAD_RIGHT':      8,
    'DPAD_UP':         5,
    'EAST':            3,
    'LEFT_SHOULDER':  11,  # left_bumper
    'LEFT_THUMB':     13,
    'NORTH':           2,
    'RIGHT_SHOULDER': 12,  # right_bumper
    'RIGHT_THUMB':    14,
    'SOUTH':           1,
    'START':           9,
    'WEST':            4,
    # LEFT_TRIGGER(9) / RIGHT_TRIGGER(16) 不映射到 slot
}
OPENP2P_MAX_BUTTONS = 6   # openp2p 每帧最多 6 个 slot
OPENP2P_NO_BUTTON   = 0   # slot 填充值

# openp2p slot 按钮顺序（与 GAMEPAD_BUTTON_ORDER 一致）
_OPENP2P_SLOT_ORDER = [
    'SOUTH', 'NORTH', 'EAST', 'WEST', 'DPAD_UP', 'DPAD_DOWN',
    'DPAD_LEFT', 'DPAD_RIGHT', 'START', 'BACK', 'LEFT_SHOULDER',
    'RIGHT_SHOULDER', 'LEFT_THUMB', 'RIGHT_THUMB',
]

# openp2p slot ID → 名称（用于 per_button_f1 输出）
_OPENP2P_ID_TO_NAME = {v: k for k, v in _NITROGEN_TO_OPENP2P_ID.items()}
OPENP2P_SLOT_IDS = sorted(_NITROGEN_TO_OPENP2P_ID.values())  # [1..14]
N_OPENP2P_BUTTONS = len(OPENP2P_SLOT_IDS)  # 14


def nitrogen_onehot_to_openp2p_slots(btn_vec: list) -> list:
    """
    将 NitroGen 21 维 one-hot 转换为 openp2p slot 编码。
    返回长度为 OPENP2P_MAX_BUTTONS 的 list，每个元素是按钮 ID（0=no_button）。
    """
    pressed_ids = []
    for nitrogen_name in _OPENP2P_SLOT_ORDER:
        nitrogen_idx = BUTTON_ACTION_TOKENS.index(nitrogen_name)
        if btn_vec[nitrogen_idx] > 0.5:
            pressed_ids.append(_NITROGEN_TO_OPENP2P_ID[nitrogen_name])
        if len(pressed_ids) >= OPENP2P_MAX_BUTTONS:
            break
    # 补齐到 6 个 slot
    while len(pressed_ids) < OPENP2P_MAX_BUTTONS:
        pressed_ids.append(OPENP2P_NO_BUTTON)
    return pressed_ids


# ── 量化函数 ──────────────────────────────────────────────────────────────

def quantize_stick(value: float) -> int:
    """连续摇杆值 [-1,1] → bin index [0, N_STICK_BINS-1]"""
    v = max(-1.0, min(1.0, float(value)))
    if abs(v) < STICK_DEADZONE:
        v = 0.0
    v01 = (v + 1.0) / 2.0
    idx = int(round(v01 * (N_STICK_BINS - 1)))
    return max(0, min(N_STICK_BINS - 1, idx))


def quantize_trigger(value: float) -> int:
    """连续扳机值 [0,1] → bin index [0, N_TRIGGER_BINS-1]"""
    v = max(0.0, min(1.0, float(value)))
    idx = int(round(v * (N_TRIGGER_BINS - 1)))
    return max(0, min(N_TRIGGER_BINS - 1, idx))


def stick_direction(bin_idx: int) -> int:
    """bin → 三分类方向：-1(负) / 0(中/死区) / +1(正)"""
    lo = STICK_CENTER - STICK_DZ_HALF
    hi = STICK_CENTER + STICK_DZ_HALF
    if bin_idx < lo:
        return -1
    elif bin_idx > hi:
        return 1
    else:
        return 0


def in_deadzone(bin_idx: int) -> bool:
    lo = STICK_CENTER - STICK_DZ_HALF
    hi = STICK_CENTER + STICK_DZ_HALF
    return lo <= bin_idx <= hi


# ── 滚动均值辅助 ──────────────────────────────────────────────────────────

class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value
        self.count += n

    @property
    def mean(self):
        return self.total / self.count if self.count > 0 else None


# ── 主计算逻辑 ────────────────────────────────────────────────────────────

def compute_metrics(input_path: str, output_path: str, mode: str = "onehot"):
    """
    mode='onehot': NitroGen 原始 16 维独立二分类
    mode='slot'  : 转换为 openp2p slot 编码后计算，与 openp2p AR 验证口径一致
    """
    assert mode in ("onehot", "slot"), f"mode 必须是 onehot 或 slot，got {mode}"

    if mode == "onehot":
        n_btn = N_VALID_BUTTONS
        btn_names = VALID_BUTTON_NAMES
    else:
        # slot 模式：6 个 slot，每个 slot 是 15 分类（0=no_button, 1-14=按钮ID）
        n_btn = OPENP2P_MAX_BUTTONS
        btn_names = [f"slot_{i}" for i in range(n_btn)]

    btn_correct   = RunningMean()
    btn_exact_match = RunningMean()
    # slot 模式下按 ID 统计 F1（14 个有效按钮 ID）
    # onehot 模式下按维度统计 F1（16 个有效维度）
    if mode == "slot":
        f1_tp = {id_: 0 for id_ in OPENP2P_SLOT_IDS}
        f1_fp = {id_: 0 for id_ in OPENP2P_SLOT_IDS}
        f1_fn = {id_: 0 for id_ in OPENP2P_SLOT_IDS}
    else:
        btn_tp = [0] * N_VALID_BUTTONS
        btn_fp = [0] * N_VALID_BUTTONS
        btn_fn = [0] * N_VALID_BUTTONS

    stick_mae       = [RunningMean() for _ in range(4)]
    stick_sq_err    = [RunningMean() for _ in range(4)]
    stick_dir_correct = [RunningMean() for _ in range(4)]
    stick_dz_correct  = [RunningMean() for _ in range(4)]
    trigger_mae     = [RunningMean() for _ in range(2)]
    trigger_binary_correct = [RunningMean() for _ in range(2)]

    total_samples = 0

    def _process_pair(pred_btn_vec, pred_jl, pred_jr, gt):
        nonlocal total_samples
        total_samples += 1

        gt_btn_vec = gt["buttons"]
        gt_jl = gt["j_left"]
        gt_jr = gt["j_right"]
        gt_lt = gt.get("left_trigger", 0.0)
        gt_rt = gt.get("right_trigger", 0.0)

        # ── 按钮指标 ────────────────────────────────────────────────────────
        if mode == "onehot":
            pred_valid = [1 if pred_btn_vec[i] > 0.5 else 0 for i in VALID_BUTTON_INDICES]
            gt_valid   = [1 if gt_btn_vec[i]   > 0.5 else 0 for i in VALID_BUTTON_INDICES]

            correct_per = [p == g for p, g in zip(pred_valid, gt_valid)]
            btn_correct.update(sum(correct_per), N_VALID_BUTTONS)
            btn_exact_match.update(1 if all(correct_per) else 0)

            for vi in range(N_VALID_BUTTONS):
                p, g = pred_valid[vi], gt_valid[vi]
                if p == 1 and g == 1: btn_tp[vi] += 1
                elif p == 1 and g == 0: btn_fp[vi] += 1
                elif p == 0 and g == 1: btn_fn[vi] += 1

        else:  # slot 模式
            pred_slots = nitrogen_onehot_to_openp2p_slots(pred_btn_vec)
            gt_slots   = nitrogen_onehot_to_openp2p_slots(gt_btn_vec)

            # slot-level accuracy（逐 slot 比较）
            correct_per = [p == g for p, g in zip(pred_slots, gt_slots)]
            btn_correct.update(sum(correct_per), OPENP2P_MAX_BUTTONS)
            btn_exact_match.update(1 if all(correct_per) else 0)

            # 按钮 ID 级别的 F1（把 slot 展开成 pressed set）
            pred_set = set(id_ for id_ in pred_slots if id_ != OPENP2P_NO_BUTTON)
            gt_set   = set(id_ for id_ in gt_slots   if id_ != OPENP2P_NO_BUTTON)
            for id_ in OPENP2P_SLOT_IDS:
                p = id_ in pred_set
                g = id_ in gt_set
                if p and g:     f1_tp[id_] += 1
                elif p and not g: f1_fp[id_] += 1
                elif not p and g: f1_fn[id_] += 1

        # ── 摇杆指标 ────────────────────────────────────────────────────────
        axes_pred = [pred_jl[0], pred_jl[1], pred_jr[0], pred_jr[1]]
        axes_gt   = [gt_jl[0],  gt_jl[1],  gt_jr[0],  gt_jr[1]]
        for i in range(4):
            p_bin = quantize_stick(axes_pred[i])
            g_bin = quantize_stick(axes_gt[i])
            ae = abs(p_bin - g_bin)
            stick_mae[i].update(ae)
            stick_sq_err[i].update(ae ** 2)
            stick_dir_correct[i].update(1 if stick_direction(p_bin) == stick_direction(g_bin) else 0)
            if in_deadzone(g_bin):
                stick_dz_correct[i].update(1 if in_deadzone(p_bin) else 0)

        # ── 扳机指标 ────────────────────────────────────────────────────────
        for i, (p_val, g_val) in enumerate([(0.0, gt_lt), (0.0, gt_rt)]):
            p_bin = quantize_trigger(p_val)
            g_bin = quantize_trigger(g_val)
            trigger_mae[i].update(abs(p_bin - g_bin))
            trigger_binary_correct[i].update(1 if (p_val > 0.0) == (g_val > 0.0) else 0)

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            action_shift = record.get("action_shift", 3)
            pred = record["pred"]
            pred_buttons_all = pred["buttons"]
            pred_jl_all      = pred["j_left"]
            pred_jr_all      = pred["j_right"]

            if "gt_sequence" in record:
                gt_sequence = record["gt_sequence"]
                for k in range(action_shift, len(gt_sequence)):
                    gt = gt_sequence[k]
                    if gt is None or k >= len(pred_buttons_all):
                        continue
                    _process_pair(pred_buttons_all[k], pred_jl_all[k], pred_jr_all[k], gt)
            elif "gt" in record:
                if action_shift >= len(pred_buttons_all):
                    continue
                _process_pair(pred_buttons_all[action_shift], pred_jl_all[action_shift],
                               pred_jr_all[action_shift], record["gt"])

    # ── 汇总 ──────────────────────────────────────────────────────────────
    result = {"total_samples": total_samples, "mode": mode}

    result["button_accuracy"]    = round(btn_correct.mean, 4) if btn_correct.mean is not None else None
    result["button_exact_match"] = round(btn_exact_match.mean, 4) if btn_exact_match.mean is not None else None
    result["button_sample_count"] = btn_correct.count

    f1_scores = []
    per_button_f1 = {}
    if mode == "onehot":
        for vi in range(N_VALID_BUTTONS):
            tp, fp, fn = btn_tp[vi], btn_fp[vi], btn_fn[vi]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
            per_button_f1[VALID_BUTTON_NAMES[vi]] = round(f1, 4)
        result["button_f1_note"] = f"macro F1 over {N_VALID_BUTTONS} valid buttons (onehot mode)"
    else:
        for id_ in OPENP2P_SLOT_IDS:
            tp, fp, fn = f1_tp[id_], f1_fp[id_], f1_fn[id_]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
            per_button_f1[_OPENP2P_ID_TO_NAME[id_]] = round(f1, 4)
        result["button_f1_note"] = f"macro F1 over {N_OPENP2P_BUTTONS} openp2p buttons (slot mode, same as AR validation)"

    result["button_f1_macro"] = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None
    result["per_button_f1"]   = per_button_f1

    # 摇杆
    axis_names = ["left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"]
    per_stick = {}
    mae_vals, rmse_vals, dir_vals, dz_vals = [], [], [], []
    for i, name in enumerate(axis_names):
        m = stick_mae[i].mean
        per_stick[f"{name}_mae"] = round(m, 4) if m is not None else None
        if m is not None:
            mae_vals.append(m)

        sq = stick_sq_err[i].mean
        rmse = math.sqrt(sq) if sq is not None else None
        per_stick[f"{name}_rmse"] = round(rmse, 4) if rmse is not None else None
        if rmse is not None:
            rmse_vals.append(rmse)

        d = stick_dir_correct[i].mean
        per_stick[f"{name}_direction_accuracy"] = round(d, 4) if d is not None else None
        if d is not None:
            dir_vals.append(d)

        dz = stick_dz_correct[i].mean
        per_stick[f"{name}_deadzone_accuracy"] = round(dz, 4) if dz is not None else None
        if dz is not None:
            dz_vals.append(dz)

    result["stick_mae"] = round(sum(mae_vals) / len(mae_vals), 4) if mae_vals else None
    result["stick_rmse"] = round(sum(rmse_vals) / len(rmse_vals), 4) if rmse_vals else None
    result["stick_direction_accuracy"] = round(sum(dir_vals) / len(dir_vals), 4) if dir_vals else None
    result["stick_deadzone_accuracy"] = round(sum(dz_vals) / len(dz_vals), 4) if dz_vals else None
    result["per_stick"] = per_stick

    # 扳机
    trigger_names = ["left_trigger", "right_trigger"]
    per_trigger = {}
    t_mae_vals, t_bin_vals = [], []
    for i, name in enumerate(trigger_names):
        m = trigger_mae[i].mean
        per_trigger[f"{name}_mae"] = round(m, 4) if m is not None else None
        if m is not None:
            t_mae_vals.append(m)

        b = trigger_binary_correct[i].mean
        per_trigger[f"{name}_binary_accuracy"] = round(b, 4) if b is not None else None
        if b is not None:
            t_bin_vals.append(b)

    result["trigger_mae"] = round(sum(t_mae_vals) / len(t_mae_vals), 4) if t_mae_vals else None
    result["trigger_binary_accuracy"] = round(sum(t_bin_vals) / len(t_bin_vals), 4) if t_bin_vals else None
    result["per_trigger"] = per_trigger

    # 输出
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"模式: {mode}  |  总样本数: {total_samples}")
    if mode == "onehot":
        print(f"\n[按钮] (16 维独立二分类)")
    else:
        print(f"\n[按钮] (openp2p slot 编码，6 slot × 15分类，与 AR 验证口径一致)")
    print(f"  accuracy:     {result['button_accuracy']}")
    print(f"  exact_match:  {result['button_exact_match']}")
    print(f"  f1_macro:     {result['button_f1_macro']}")
    print(f"\n[摇杆] (量化 bin, N={N_STICK_BINS})")
    print(f"  mae:               {result['stick_mae']}")
    print(f"  rmse:              {result['stick_rmse']}")
    print(f"  direction_acc:     {result['stick_direction_accuracy']}")
    print(f"  deadzone_acc:      {result['stick_deadzone_accuracy']}")
    print(f"\n[扳机] (量化 bin, N={N_TRIGGER_BINS})")
    print(f"  mae:               {result['trigger_mae']}")
    print(f"  binary_accuracy:   {result['trigger_binary_accuracy']}")
    print(f"\n结果已保存至: {output_path}")
    print(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(description="计算 NitroGen eval.jsonl 的 openp2p gamepad 指标")
    parser.add_argument("--input", "-i", default="NitroGen_model_cuphead_results/eval.jsonl")
    parser.add_argument("--output", "-o", default="NitroGen_model_cuphead_results/metrics.json")
    parser.add_argument(
        "--mode", choices=["onehot", "slot"], default="onehot",
        help="onehot: 16维独立二分类（默认）; slot: 转换为openp2p slot编码，与AR验证口径一致"
    )
    args = parser.parse_args()
    compute_metrics(args.input, args.output, mode=args.mode)


if __name__ == "__main__":
    main()
