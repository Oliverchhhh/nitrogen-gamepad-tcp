"""
Autoregressive Validation — 逐帧 KV cache 自回归验证，贴近真实推理。

与 teacher-forcing validation 的区别：
  - teacher-forcing: 每步输入 GT action，评估"给定完美历史的预测能力"
  - autoregressive:  每步输入模型自己采样的 action，评估"实际推理时的表现"

使用 online_kv_cache_predict 逐帧推理（和 inference.py 完全一致的路径），
将采样的 action 与 GT 对比计算指标。

用法:
  python elefant/policy_model/validation_autoregressive.py \
    --checkpoint_path output/.../checkpoint-step=00040000.ckpt \
    --config_path config/policy_model/150M_local_nitrogen_dataset_current.yaml \
    --data_folder NitroGen_cuphead_toy \
    [--n_sequences 50] [--sampling_temperature 1.0]
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
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.stage3_finetune import (
    Stage3DataModule,
    Stage3FutureVisionLightning,
    Stage3LabelledBCLightning,
)
from elefant.policy_model.validation import (
    move_action_label_video_dataset_item_to_device,
    set_validation_dataset_cfg_to_single_thread,
    _is_future_vision_model,
)


# ---------------------------------------------------------------------------
# 自回归专用指标（直接对比采样 action index vs GT label）
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


class AutoregressiveGamepadMetrics:
    """
    自回归验证指标。直接对比采样的 action index tensor vs GT label tensor。

    采样结果 shape: [n_actions] (每帧一个 int tensor)
    GT label shape:  [n_actions]

    指标与 GamepadActionMetrics 一致，但输入是 index 而非 logits。
    """

    def __init__(self, n_buttons: int, n_stick_bins: int, n_trigger_bins: int,
                 stick_deadzone_half: int = 1):
        self.n_buttons = n_buttons
        self.n_stick_bins = n_stick_bins
        self.n_trigger_bins = n_trigger_bins
        self.stick_center = n_stick_bins // 2
        self.stick_dz_half = stick_deadzone_half

        # 按钮
        self.button_correct = _RunningMean()
        self.button_exact_match = _RunningMean()
        self._btn_tp = [0] * n_buttons
        self._btn_fp = [0] * n_buttons
        self._btn_fn = [0] * n_buttons

        # 摇杆 (4 轴)
        self.stick_mae = [_RunningMean() for _ in range(4)]
        self.stick_sq_err = [_RunningMean() for _ in range(4)]
        self.stick_dir_correct = [_RunningMean() for _ in range(4)]
        self.stick_dz_correct = [_RunningMean() for _ in range(4)]

        # 扳机 (2 轴)
        self.trigger_mae = [_RunningMean() for _ in range(2)]
        self.trigger_binary_correct = [_RunningMean() for _ in range(2)]

    @staticmethod
    def _to_direction(val: int, center: int, dz: int) -> int:
        if val < center - dz:
            return -1
        elif val > center + dz:
            return 1
        return 0

    def update_step(self, pred_action: torch.Tensor, gt_action: torch.Tensor, n_buttons: int):
        """
        更新单帧指标。
        pred_action: [n_actions] — 模型采样的 action index
        gt_action:   [n_actions] — GT label（可能含 -100）
        """
        k = n_buttons

        # ---- 按钮 ----
        pred_btn = pred_action[:k]
        gt_btn = gt_action[:k]
        valid = gt_btn != -100

        if valid.any():
            correct = (pred_btn[valid] == gt_btn[valid])
            self.button_correct.update(correct.float().sum().item(), correct.numel())

            if valid.all():
                em = (pred_btn == gt_btn).all().float().item()
                self.button_exact_match.update(em, 1)

            for bi in range(min(k, self.n_buttons)):
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

        # ---- 摇杆 (4 轴) ----
        for i in range(4):
            col = k + i
            if gt_action[col] == -100:
                continue
            p = pred_action[col].float()
            g = gt_action[col].float()
            ae = (p - g).abs().item()
            self.stick_mae[i].update(ae, 1)
            self.stick_sq_err[i].update(ae ** 2, 1)

            d_pred = self._to_direction(pred_action[col].item(), self.stick_center, self.stick_dz_half)
            d_gt = self._to_direction(gt_action[col].item(), self.stick_center, self.stick_dz_half)
            self.stick_dir_correct[i].update(float(d_pred == d_gt), 1)

            in_dz_gt = (self.stick_center - self.stick_dz_half
                        <= gt_action[col].item()
                        <= self.stick_center + self.stick_dz_half)
            if in_dz_gt:
                in_dz_pred = (self.stick_center - self.stick_dz_half
                              <= pred_action[col].item()
                              <= self.stick_center + self.stick_dz_half)
                self.stick_dz_correct[i].update(float(in_dz_pred), 1)

        # ---- 扳机 (2 轴) ----
        for i in range(2):
            col = k + 4 + i
            if gt_action[col] == -100:
                continue
            p = pred_action[col].float()
            g = gt_action[col].float()
            ae = (p - g).abs().item()
            self.trigger_mae[i].update(ae, 1)

            bp = (pred_action[col] > 0).item()
            bg = (gt_action[col] > 0).item()
            self.trigger_binary_correct[i].update(float(bp == bg), 1)

    def compute(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        # 按钮
        result["button_accuracy"] = self.button_correct.mean
        result["button_exact_match"] = self.button_exact_match.mean
        result["button_sample_count"] = self.button_correct.count

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

        # 摇杆
        axis_names = ["left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"]
        per_stick = {}
        mae_vals, rmse_vals, dir_vals, dz_vals = [], [], [], []
        for i, name in enumerate(axis_names):
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

        # 扳机
        trigger_names = ["left_trigger", "right_trigger"]
        per_trigger = {}
        t_mae_vals, t_bin_vals = [], []
        for i, name in enumerate(trigger_names):
            m = self.trigger_mae[i].mean
            per_trigger[f"{name}_mae"] = round(m, 4) if m is not None else None
            if m is not None:
                t_mae_vals.append(m)

            b = self.trigger_binary_correct[i].mean
            per_trigger[f"{name}_binary_accuracy"] = round(b, 4) if b is not None else None
            if b is not None:
                t_bin_vals.append(b)

        result["trigger_mae"] = round(sum(t_mae_vals) / len(t_mae_vals), 4) if t_mae_vals else None
        result["trigger_binary_accuracy"] = round(sum(t_bin_vals) / len(t_bin_vals), 4) if t_bin_vals else None
        result["per_trigger"] = per_trigger

        return result


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def _build_gamepad_action_sampler(model, sampling_temperature: float):
    """构建和 online_kv_cache_predict 内部一致的 action_sampler 闭包（备用）"""
    def _action_sampler(action_token, action_idx, sampled_actions):
        return model._gamepad_action_sampler(
            action_token, action_idx, sampled_actions,
            sampling_temperature, None,
        )
    return _action_sampler


def _load_model(checkpoint_path: str, config: LightningPolicyConfig):
    if _is_future_vision_model(config):
        logging.info("Loading Stage3FutureVisionLightning for AR validation")
        model = Stage3FutureVisionLightning(config=config, inference_mode=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model
    logging.info("Loading Stage3LabelledBCLightning for AR validation")
    return Stage3LabelledBCLightning.load_from_checkpoint(
        checkpoint_path, config=config, inference_mode=True
    )


def run_autoregressive_validation(
    checkpoint_path: str,
    config_path: str,
    data_folder: Optional[str] = None,
    n_sequences: int = 50,
    sampling_temperature: float = 1.0,
):
    """
    自回归验证：使用 online_kv_cache_predict 逐帧推理（和 inference.py 完全一致），
    将采样的 action 与 GT 对比。

    每个视频序列：
      1. 初始化 KV cache
      2. 逐帧: frame[t] → online_kv_cache_predict → sampled_action
      3. sampled_action vs GT action → 计算指标
      4. sampled_action 自动作为下一步的 KV cache 历史
    """
    t0 = time.time()
    config = load_config(config_path, LightningPolicyConfig)

    if data_folder:
        for ds in config.stage3_finetune.validation_datasets:
            ds.local_prefix = data_folder
        config.stage3_finetune.training_dataset.local_prefix = data_folder

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
    model_dtype = next(model.parameters()).dtype

    if hasattr(model, "bc_transformer") and hasattr(
        model.bc_transformer, "block_mask_to_device"
    ):
        model.bc_transformer.block_mask_to_device(model.device)

    # 初始化 KV cache（和 inference.py 一致）
    max_virtual_idx = config.shared.n_seq_timesteps * model.bc_transformer.step_size
    model.bc_transformer.rebuild_rope_cache(max_virtual_idx)
    model.bc_transformer.setup_kv_cache(batch_size=1, device=model.device)

    # action_decoder 的 KV cache 在 __init__ 时创建在 CPU 上，
    # model.to("cuda") 后需要重新初始化到 GPU
    if hasattr(model.bc_transformer, 'action_decoder'):
        model.bc_transformer.action_decoder._setup_kv_cache_state()

    # GamepadAutoregressiveActionMapping.make_empty_action 硬编码 device=cpu，
    # 在 compile=False 模式下会导致设备不匹配。Patch 为使用 cuda。
    if hasattr(model, 'action_mapping'):
        _orig_make_empty = model.action_mapping.make_empty_action
        _device = model.device
        def _make_empty_action_cuda(T: int):
            return _orig_make_empty(T).to(_device)
        model.action_mapping.make_empty_action = _make_empty_action_cuda

    is_gamepad = hasattr(model, '_use_gamepad_mapping') and model._use_gamepad_mapping()
    if not is_gamepad:
        raise NotImplementedError("AR validation 目前仅支持手柄模式")

    n_buttons = model.n_gamepad_button_actions
    n_actions = model.n_actions

    datamodule = Stage3DataModule(config)
    datamodule.setup("stage3_finetune")
    val_dataloaders = datamodule.val_dataloader()
    del datamodule

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"n_buttons={n_buttons}, n_actions={n_actions}, temperature={sampling_temperature}")

    all_results: Dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "config_path": config_path,
        "mode": "autoregressive",
        "sampling_temperature": sampling_temperature,
    }

    for val_set_name, val_dataloader in val_dataloaders.items():
        print(f"\n{'='*60}")
        print(f"AR Validation: {val_set_name}")
        print(f"{'='*60}")

        gp_metrics = AutoregressiveGamepadMetrics(
            n_buttons=n_buttons,
            n_stick_bins=model.n_gamepad_stick_bins,
            n_trigger_bins=model.n_gamepad_trigger_bins,
        )

        seq_count = 0
        for batch_idx, batch in enumerate(val_dataloader):
            if seq_count >= n_sequences:
                break

            batch_to_cuda = move_action_label_video_dataset_item_to_device(
                batch, "cuda", dtype=model_dtype
            )

            B = batch_to_cuda.frames.shape[0]
            T = batch_to_cuda.frames.shape[1]
            frames = batch_to_cuda.frames          # [B, T, C, H, W]
            gt_actions = batch_to_cuda.action_annotations  # [B, T, n_actions]
            text_embed = batch_to_cuda.text_embeddings

            # 逐样本处理（B=1）
            for b in range(B):
                if seq_count >= n_sequences:
                    break

                # 初始化 KV cache
                kv_cache_state = model.bc_transformer.init_kv_cache_state()
                idx = torch.tensor(0, dtype=torch.int64, device="cuda")

                for t in range(T):
                    # online_kv_cache_predict 内部会 _normalize_frames + unsqueeze
                    frame_t = frames[b, t]  # [C, H, W]

                    # text embedding for this step
                    text_t = text_embed[b, t:t+1].unsqueeze(0) if text_embed is not None else None

                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        sampled_action, idx, kv_cache_state = model.online_kv_cache_predict(
                            frame_t,
                            idx=idx,
                            kv_cache_state=kv_cache_state,
                            sampling_temperature=sampling_temperature,
                            text_tokens_embed=text_t,
                            compile=True,
                        )

                    # sampled_action: [n_actions] int tensor
                    gt_action_t = gt_actions[b, t]  # [n_actions]

                    gp_metrics.update_step(
                        pred_action=sampled_action.squeeze(),
                        gt_action=gt_action_t,
                        n_buttons=n_buttons,
                    )

                seq_count += 1
                if seq_count % 5 == 0 or seq_count >= n_sequences:
                    elapsed = time.time() - t0
                    print(f"  Processed {seq_count}/{n_sequences} sequences ({elapsed:.1f}s)")

        # 汇总
        gp_result = gp_metrics.compute()

        print(f"\n{'='*60}")
        print(f"[{val_set_name}] AR Gamepad Metrics (temperature={sampling_temperature}):")
        print(f"{'='*60}")
        print(f"  Button accuracy:        {gp_result.get('button_accuracy')}")
        print(f"  Button exact match:     {gp_result.get('button_exact_match')}")
        print(f"  Button F1 (macro):      {gp_result.get('button_f1_macro')}")
        print(f"  Stick MAE (bin):        {gp_result.get('stick_mae')}")
        print(f"  Stick RMSE (bin):       {gp_result.get('stick_rmse')}")
        print(f"  Stick direction acc:    {gp_result.get('stick_direction_accuracy')}")
        print(f"  Stick deadzone acc:     {gp_result.get('stick_deadzone_accuracy')}")
        print(f"  Trigger MAE (bin):      {gp_result.get('trigger_mae')}")
        print(f"  Trigger binary acc:     {gp_result.get('trigger_binary_accuracy')}")
        print(f"{'='*60}")

        all_results[val_set_name] = {"gamepad_metrics": gp_result}

    total_time = time.time() - t0
    all_results["total_validation_time_sec"] = round(total_time, 3)
    all_results["n_sequences"] = seq_count

    # 保存 JSON
    output_dir = os.path.dirname(checkpoint_path)
    step_str = os.path.basename(checkpoint_path).split("step=")[-1].split(".")[0]
    json_path = os.path.join(output_dir, f"validation_ar_step{step_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAR validation results saved to: {json_path}")
    print(f"Total time: {total_time:.1f}s")

    # 显式释放 GPU 显存，避免僵尸进程
    del model
    torch.cuda.empty_cache()
    print("GPU memory released.")

    return all_results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Autoregressive validation (KV cache)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, default=None)
    parser.add_argument("--n_sequences", type=int, default=50,
                        help="验证的视频序列数量")
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    args = parser.parse_args()

    run_autoregressive_validation(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        data_folder=args.data_folder,
        n_sequences=args.n_sequences,
        sampling_temperature=args.sampling_temperature,
    )


if __name__ == "__main__":
    main()
