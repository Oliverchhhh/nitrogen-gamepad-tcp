"""
Autoregressive Validation — direct action paradigm (non-autoregressive decoder).

与 teacher-forcing (validation_action_direct.py) 的区别：
  - teacher-forcing: 每步输入 GT action，评估"给定完美历史的预测能力"
  - autoregressive:  每步输入模型自己采样的 action，评估"实际推理时的表现"

使用 online_kv_cache_predict 逐帧推理（和 inference.py 完全一致的路径），
direct head 走 sigmoid/argmax 采样（无温度参数，确定性解码）。

用法:
  python elefant/policy_model/validation_autoregressive_action_direct.py \\
    --checkpoint_path output/.../checkpoint-step=00040000.ckpt \\
    --config_path config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml \\
    --data_folder cuphead_dataset_converted \\
    [--n_sequences 50]
"""

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import torch

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig, ValidationDatasetConfig
from elefant.policy_model.stage3_finetune import (
    Stage3DataModule,
    Stage3FutureVisionLightning,
    Stage3LabelledBCLightning,
)
from elefant.policy_model.validation_action_direct import (
    move_batch_to_device,
    set_validation_dataset_cfg_to_single_thread,
)


# ---------------------------------------------------------------------------
# 自回归专用指标 — direct 范式（button binary + stick CE，无扳机）
# ---------------------------------------------------------------------------

@dataclass
class _RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, sum_value: float, n: int = 1):
        self.total += sum_value
        self.count += n

    @property
    def mean(self) -> Optional[float]:
        return self.total / self.count if self.count > 0 else None


class AutoregressiveDirectMetrics:
    """
    自回归验证指标 — direct action 范式。

    输入均为 index tensor（非 logits），每帧调用一次 update_step()。

    按钮 (binary BCE 范式):
        button_accuracy      per-button 准确率（pred > 0 vs gt > 0）
        button_exact_match   一帧内所有按钮全部正确的比例
        button_f1_macro      宏平均 F1

    摇杆 (4 轴 CE 范式):
        stick_mae            bin MAE（4 轴平均）
        stick_rmse           bin RMSE
        stick_direction_acc  三分类方向准确率
        stick_deadzone_acc   GT 在死区时预测也在死区的比例

    注：direct 范式无扳机 head，不计算 trigger 指标。
    """

    STICK_AXES = ["left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"]

    def __init__(self, n_buttons: int, n_stick_bins: int):
        self.n_buttons = n_buttons
        self.n_stick_bins = n_stick_bins
        self.stick_center = n_stick_bins // 2
        self.stick_dz_half = 0 if n_stick_bins <= 3 else 1

        # buttons
        self.btn_correct = _RunningMean()
        self.btn_exact_match = _RunningMean()
        self._btn_tp = [0] * n_buttons
        self._btn_fp = [0] * n_buttons
        self._btn_fn = [0] * n_buttons

        # sticks
        self.stick_mae = [_RunningMean() for _ in range(4)]
        self.stick_sq_err = [_RunningMean() for _ in range(4)]
        self.stick_dir_correct = [_RunningMean() for _ in range(4)]
        self.stick_dz_correct = [_RunningMean() for _ in range(4)]

    @staticmethod
    def _to_direction(val: int, center: int, dz: int) -> int:
        if val < center - dz:
            return -1
        elif val > center + dz:
            return 1
        return 0

    def update_step(self, pred_action: torch.Tensor, gt_action: torch.Tensor):
        """
        更新单帧指标。
        pred_action: [n_buttons + 4]  — 模型采样的 action index（direct head 输出）
        gt_action:   [n_buttons + 4]  — GT label（可能含 -100）
        """
        k = self.n_buttons

        # ---- buttons (binary) ----
        pred_btn = pred_action[:k]
        gt_btn = gt_action[:k]
        valid = gt_btn != -100

        if valid.any():
            # direct head 输出已经是 0/1，gt 也是 0/1
            correct = pred_btn[valid] == gt_btn[valid]
            self.btn_correct.update(correct.float().sum().item(), correct.numel())

            if valid.all():
                em = (pred_btn == gt_btn).all().float().item()
                self.btn_exact_match.update(em, 1)

            for bi in range(self.n_buttons):
                if gt_btn[bi] == -100:
                    continue
                p = (pred_btn[bi] > 0).item()
                g = (gt_btn[bi] > 0).item()
                if p and g:
                    self._btn_tp[bi] += 1
                elif p and not g:
                    self._btn_fp[bi] += 1
                elif not p and g:
                    self._btn_fn[bi] += 1

        # ---- sticks (4 axes) ----
        for i in range(4):
            col = k + i
            if col >= gt_action.shape[0] or gt_action[col] == -100:
                continue
            p_val = pred_action[col].float()
            g_val = gt_action[col].float()
            ae = (p_val - g_val).abs().item()
            self.stick_mae[i].update(ae, 1)
            self.stick_sq_err[i].update(ae ** 2, 1)

            d_pred = self._to_direction(pred_action[col].item(), self.stick_center, self.stick_dz_half)
            d_gt = self._to_direction(gt_action[col].item(), self.stick_center, self.stick_dz_half)
            self.stick_dir_correct[i].update(float(d_pred == d_gt), 1)

            in_dz_gt = (
                self.stick_center - self.stick_dz_half
                <= gt_action[col].item()
                <= self.stick_center + self.stick_dz_half
            )
            if in_dz_gt:
                in_dz_pred = (
                    self.stick_center - self.stick_dz_half
                    <= pred_action[col].item()
                    <= self.stick_center + self.stick_dz_half
                )
                self.stick_dz_correct[i].update(float(in_dz_pred), 1)

    def compute(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        # buttons
        result["button_accuracy"] = self.btn_correct.mean
        result["button_exact_match"] = self.btn_exact_match.mean
        result["button_sample_count"] = self.btn_correct.count

        f1_scores = []
        per_button_f1 = {}
        for bi in range(self.n_buttons):
            tp, fp, fn = self._btn_tp[bi], self._btn_fp[bi], self._btn_fn[bi]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
            per_button_f1[f"button_{bi}_f1"] = round(f1, 4)
        result["button_f1_macro"] = sum(f1_scores) / len(f1_scores) if f1_scores else None
        result["per_button_f1"] = per_button_f1

        # sticks
        per_stick = {}
        mae_vals, rmse_vals, dir_vals, dz_vals = [], [], [], []
        for i, name in enumerate(self.STICK_AXES):
            m = self.stick_mae[i].mean
            per_stick[f"{name}_mae"] = round(m, 4) if m is not None else None
            if m is not None:
                mae_vals.append(m)

            sq = self.stick_sq_err[i].mean
            rmse = math.sqrt(sq) if sq is not None else None
            per_stick[f"{name}_rmse"] = round(rmse, 4) if rmse is not None else None
            if rmse is not None:
                rmse_vals.append(rmse)

            d = self.stick_dir_correct[i].mean
            per_stick[f"{name}_direction_accuracy"] = round(d, 4) if d is not None else None
            if d is not None:
                dir_vals.append(d)

            dz = self.stick_dz_correct[i].mean
            per_stick[f"{name}_deadzone_accuracy"] = round(dz, 4) if dz is not None else None
            if dz is not None:
                dz_vals.append(dz)

        result["stick_mae"] = round(sum(mae_vals) / len(mae_vals), 4) if mae_vals else None
        result["stick_rmse"] = round(sum(rmse_vals) / len(rmse_vals), 4) if rmse_vals else None
        result["stick_direction_accuracy"] = round(sum(dir_vals) / len(dir_vals), 4) if dir_vals else None
        result["stick_deadzone_accuracy"] = round(sum(dz_vals) / len(dz_vals), 4) if dz_vals else None
        result["per_stick"] = per_stick

        return result


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, config: LightningPolicyConfig):
    has_future = (
        config.stage3_finetune.state_target_tokenizer is not None
        or config.stage3_finetune.use_precomputed_vision_features
    )
    if has_future:
        logging.info("Loading Stage3FutureVisionLightning for AR-direct validation")
        model = Stage3FutureVisionLightning(config=config, inference_mode=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
        if incompatible.missing_keys:
            logging.warning("Missing keys: %s", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            logging.warning("Unexpected keys: %s", incompatible.unexpected_keys)
        return model
    logging.info("Loading Stage3LabelledBCLightning for AR-direct validation")
    return Stage3LabelledBCLightning.load_from_checkpoint(
        checkpoint_path, config=config, inference_mode=True
    )


def _ensure_validation_datasets(
    config: LightningPolicyConfig,
    data_folder: Optional[str],
) -> None:
    """
    Ensure config.stage3_finetune.validation_datasets is non-empty.

    Some local configs only define training_dataset (or legacy validation_dataset),
    while AR validation expects validation_datasets.
    """
    stage3 = config.stage3_finetune
    if len(stage3.validation_datasets) == 0:
        # Fallback: build a single validation dataset from training dataset fields.
        fallback = stage3.training_dataset.model_dump()
        fallback["validation_name"] = "validation_0"
        stage3.validation_datasets = [ValidationDatasetConfig(**fallback)]

    for i, ds in enumerate(stage3.validation_datasets):
        if data_folder:
            ds.local_prefix = data_folder
        if not ds.validation_name:
            ds.validation_name = f"validation_{i}"


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def run_autoregressive_direct_validation(
    checkpoint_path: str,
    config_path: str,
    data_folder: Optional[str] = None,
    n_sequences: int = 50,
):
    """
    自回归验证 — direct action 范式。

    每个视频序列：
      1. 初始化 KV cache
      2. 逐帧: frame[t] → online_kv_cache_predict (direct_head_fn) → sampled_action
      3. sampled_action vs GT action → 计算指标
      4. sampled_action 自动写入 KV cache 历史（direct_action_in_proj 路径）
    """
    t0 = time.time()
    config = load_config(config_path, LightningPolicyConfig)

    if data_folder:
        config.stage3_finetune.training_dataset.local_prefix = data_folder
    _ensure_validation_datasets(config, data_folder)

    BATCH_SIZE = 1
    for idx in range(len(config.stage3_finetune.validation_datasets)):
        config.stage3_finetune.validation_datasets[idx] = (
            set_validation_dataset_cfg_to_single_thread(
                config.stage3_finetune.validation_datasets[idx],
                batch_size=BATCH_SIZE,
            )
        )

    model = _load_model(checkpoint_path, config)
    model.eval()
    model = model.to("cuda")

    assert model._use_gamepad_direct_mapping(), (
        "This script requires action_mapping_type='gamepad_direct'. "
        "Use validation_autoregressive.py for other mapping types."
    )

    if hasattr(model, "bc_transformer") and hasattr(
        model.bc_transformer, "block_mask_to_device"
    ):
        model.bc_transformer.block_mask_to_device(model.device)

    # KV cache 初始化（和 inference.py 一致）
    max_virtual_idx = config.shared.n_seq_timesteps * model.bc_transformer.step_size
    model.bc_transformer.rebuild_rope_cache(max_virtual_idx)
    model.bc_transformer.setup_kv_cache(batch_size=1, device=model.device)

    # direct 模式无 action_decoder，但 make_empty_action 仍需在 cuda 上
    if hasattr(model, "action_mapping"):
        _orig_make_empty = model.action_mapping.make_empty_action
        _device = model.device
        def _make_empty_action_cuda(T: int):
            return _orig_make_empty(T).to(_device)
        model.action_mapping.make_empty_action = _make_empty_action_cuda

    n_buttons = model.n_direct_buttons
    n_stick_bins = model.n_direct_stick_bins

    datamodule = Stage3DataModule(config)
    datamodule.setup("stage3_finetune")
    val_dataloaders = datamodule.val_dataloader()
    del datamodule
    if len(val_dataloaders) == 0:
        raise RuntimeError(
            "No validation dataloaders were created. "
            "Please check validation_datasets/local_prefix in config."
        )

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"n_direct_buttons={n_buttons}, n_direct_stick_bins={n_stick_bins}")

    all_results: Dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "config_path": config_path,
        "mode": "autoregressive_direct",
    }
    total_seq_count = 0

    for val_set_name, val_dataloader in val_dataloaders.items():
        print(f"\n{'='*60}")
        print(f"AR-Direct Validation: {val_set_name}")
        print(f"{'='*60}")

        ar_metrics = AutoregressiveDirectMetrics(
            n_buttons=n_buttons,
            n_stick_bins=n_stick_bins,
        )

        seq_count = 0
        for batch_idx, batch in enumerate(val_dataloader):
            if seq_count >= n_sequences:
                break

            batch_cuda = move_batch_to_device(batch, "cuda")
            B = batch_cuda.frames.shape[0]
            T = batch_cuda.frames.shape[1]
            frames = batch_cuda.frames           # [B, T, C, H, W]
            gt_actions = batch_cuda.action_annotations  # [B, T, n_actions]
            text_embed = batch_cuda.text_embeddings

            for b in range(B):
                if seq_count >= n_sequences:
                    break

                kv_cache_state = model.bc_transformer.init_kv_cache_state()
                idx = torch.tensor(0, dtype=torch.int64, device="cuda")

                for t in range(T):
                    frame_t = frames[b, t]  # [C, H, W]
                    text_t = (
                        text_embed[b, t:t+1].unsqueeze(0)
                        if text_embed is not None else None
                    )

                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        sampled_action, idx, kv_cache_state = model.online_kv_cache_predict(
                            frame_t,
                            idx=idx,
                            kv_cache_state=kv_cache_state,
                            text_tokens_embed=text_t,
                            compile=True,
                        )

                    # sampled_action: [1, n_buttons+4]  → squeeze to [n_buttons+4]
                    pred = sampled_action.squeeze(0)   # [n_buttons+4]
                    gt = gt_actions[b, t]              # [n_actions] — may be longer (no trigger cols)

                    # gt_actions 列顺序: [buttons..., lx, ly, rx, ry, lt, rt]
                    # direct pred 只有 [buttons..., lx, ly, rx, ry]，取前 n_buttons+4 列对齐
                    gt_direct = gt[: n_buttons + 4]

                    ar_metrics.update_step(pred_action=pred, gt_action=gt_direct)

                seq_count += 1
                total_seq_count += 1
                if seq_count % 5 == 0 or seq_count >= n_sequences:
                    elapsed = time.time() - t0
                    print(f"  Processed {seq_count}/{n_sequences} sequences ({elapsed:.1f}s)")

        result = ar_metrics.compute()

        print(f"\n{'='*60}")
        print(f"[{val_set_name}] AR-Direct Metrics:")
        print(f"{'='*60}")
        print(f"  Button accuracy:        {result.get('button_accuracy')}")
        print(f"  Button exact match:     {result.get('button_exact_match')}")
        print(f"  Button F1 (macro):      {result.get('button_f1_macro')}")
        print(f"  Stick MAE (bin):        {result.get('stick_mae')}")
        print(f"  Stick RMSE (bin):       {result.get('stick_rmse')}")
        print(f"  Stick direction acc:    {result.get('stick_direction_accuracy')}")
        print(f"  Stick deadzone acc:     {result.get('stick_deadzone_accuracy')}")
        print(f"{'='*60}")

        all_results[val_set_name] = {"direct_action_metrics": result}

    total_time = time.time() - t0
    all_results["total_validation_time_sec"] = round(total_time, 3)
    all_results["n_sequences"] = total_seq_count

    # 保存 JSON
    output_dir = os.path.dirname(checkpoint_path)
    step_str = os.path.basename(checkpoint_path).split("step=")[-1].split(".")[0]
    json_path = os.path.join(output_dir, f"validation_ar_direct_step{step_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAR-direct validation results saved to: {json_path}")
    print(f"Total time: {total_time:.1f}s")

    del model
    torch.cuda.empty_cache()
    print("GPU memory released.")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Autoregressive validation — direct action paradigm")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, default=None,
                        help="覆盖 config 中的 local_prefix（数据集根目录）")
    parser.add_argument("--n_sequences", type=int, default=50,
                        help="验证的视频序列数量")
    args = parser.parse_args()

    run_autoregressive_direct_validation(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        data_folder=args.data_folder,
        n_sequences=args.n_sequences,
    )


if __name__ == "__main__":
    main()
