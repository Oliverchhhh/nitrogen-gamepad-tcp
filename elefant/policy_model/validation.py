"""
Example:
uv run elefant/policy_model/validation.py \
  --checkpoint_dir=output/policy_model/150M_nitrogen/stage3_finetune \
  --config_path=config/policy_model/150M_local_nitrogen_dataset_current.yaml
"""

import re
import argparse
import json
import os
import wandb
import logging
import shutil
import torch
import time

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.stage3_finetune import (
    Stage3DataModule,
    Stage3LabelledBCLightning,
    Stage3FutureVisionLightning,
    GamepadActionLogits,
)
from elefant.data import ActionLabelVideoDatasetItem, StructuredAction
from elefant.metrics import LossMetric
from elefant.torch import cross_entropy_to_perplexity


# ---------------------------------------------------------------------------
# Gamepad Action Metrics — 按钮 / 摇杆 / 扳机 分类评估
# ---------------------------------------------------------------------------

@dataclass
class _RunningMean:
    """轻量在线均值累加器。
    调用方式: update(sum_value, count) — 传入一批样本的总和与样本数。
    """
    total: float = 0.0
    count: int = 0

    def update(self, sum_value: float, n: int = 1):
        self.total += sum_value
        self.count += n

    @property
    def mean(self) -> Optional[float]:
        return self.total / self.count if self.count > 0 else None


class GamepadActionMetrics:
    """
    手柄动作评估指标集合。

    在 validation loop 中逐 batch 调用 update()，最后调用 compute() 得到所有指标。
    所有指标均在 CPU 上计算，避免占用 GPU 显存。

    指标清单:
      按钮:
        - button_accuracy          每按钮逐帧准确率
        - button_exact_match       一帧内所有按钮全部正确的比例
        - button_f1_macro          宏平均 F1（按钮级别）
      摇杆:
        - stick_mae                bin MAE（4 轴平均）
        - stick_rmse               bin RMSE（4 轴平均）
        - stick_direction_accuracy 三分类方向准确率（左/中/右 或 下/中/上）
        - stick_deadzone_accuracy  GT 在死区时预测也在死区的比例
      扳机:
        - trigger_mae              bin MAE（2 轴平均）
        - trigger_binary_accuracy  按下/未按 二分类准确率
    """

    def __init__(self, n_buttons: int, n_stick_bins: int, n_trigger_bins: int,
                 stick_deadzone_half: int = 1):
        self.n_buttons = n_buttons
        self.n_stick_bins = n_stick_bins
        self.n_trigger_bins = n_trigger_bins
        self.stick_center = n_stick_bins // 2
        self.stick_dz_half = stick_deadzone_half  # center ± dz_half 视为死区

        # 按钮
        self.button_correct = _RunningMean()
        self.button_exact_match = _RunningMean()
        # per-button TP/FP/FN for F1
        self._btn_tp = [0] * n_buttons
        self._btn_fp = [0] * n_buttons
        self._btn_fn = [0] * n_buttons

        # 摇杆（4 轴: lx, ly, rx, ry）
        self.stick_mae = [_RunningMean() for _ in range(4)]
        self.stick_sq_err = [_RunningMean() for _ in range(4)]
        self.stick_dir_correct = [_RunningMean() for _ in range(4)]
        self.stick_dz_correct = [_RunningMean() for _ in range(4)]

        # 扳机（2 轴: lt, rt）
        self.trigger_mae = [_RunningMean() for _ in range(2)]
        self.trigger_binary_correct = [_RunningMean() for _ in range(2)]

    # ---- 内部工具 ----
    @staticmethod
    def _to_direction(bin_idx: torch.Tensor, center: int, dz: int) -> torch.Tensor:
        """将 bin 索引映射为 {-1, 0, 1} 三分类"""
        d = torch.zeros_like(bin_idx)
        d[bin_idx < center - dz] = -1
        d[bin_idx > center + dz] = 1
        return d

    def update(self, action_logits, masked_labels: torch.Tensor, n_buttons: int):
        """
        Args:
            action_logits: GamepadActionLogits (NamedTuple)
            masked_labels: [B, T, n_actions]  (ignore_index = -100)
            n_buttons: 按钮动作数量 (= model.n_gamepad_button_actions)
        """
        k = n_buttons

        # ---- 按钮 ----
        # buttons logits: [B, T, k, n_choices]
        pred_btn = action_logits.buttons.argmax(dim=-1)  # [B, T, k]
        gt_btn = masked_labels[:, :, :k]                 # [B, T, k]
        valid_btn = gt_btn != -100

        if valid_btn.any():
            correct = (pred_btn[valid_btn] == gt_btn[valid_btn])
            self.button_correct.update(correct.float().sum().item(), correct.numel())

            # exact match: 每帧所有按钮都对
            frame_valid = valid_btn.all(dim=-1)  # [B, T]
            if frame_valid.any():
                frame_correct = (pred_btn == gt_btn).all(dim=-1)  # [B, T]
                em = frame_correct[frame_valid]
                self.button_exact_match.update(em.float().sum().item(), em.numel())

            # per-button F1 (将 label > 0 视为"按下")
            for bi in range(min(k, self.n_buttons)):
                v = valid_btn[:, :, bi]
                if not v.any():
                    continue
                p = (pred_btn[:, :, bi][v] > 0)
                g = (gt_btn[:, :, bi][v] > 0)
                self._btn_tp[bi] += (p & g).sum().item()
                self._btn_fp[bi] += (p & ~g).sum().item()
                self._btn_fn[bi] += (~p & g).sum().item()

        # ---- 摇杆 (4 轴) ----
        stick_logits = [
            action_logits.left_stick_x,   # [B, T, 1, n_stick_bins]
            action_logits.left_stick_y,
            action_logits.right_stick_x,
            action_logits.right_stick_y,
        ]
        for i, logit in enumerate(stick_logits):
            col = k + i
            pred = logit.argmax(dim=-1).squeeze(-1)  # [B, T]
            gt = masked_labels[:, :, col]
            valid = gt != -100
            if not valid.any():
                continue
            p, g = pred[valid].float(), gt[valid].float()
            ae = (p - g).abs()
            self.stick_mae[i].update(ae.sum().item(), ae.numel())
            self.stick_sq_err[i].update((ae ** 2).sum().item(), ae.numel())

            # 方向三分类
            d_pred = self._to_direction(pred[valid], self.stick_center, self.stick_dz_half)
            d_gt = self._to_direction(gt[valid], self.stick_center, self.stick_dz_half)
            dc = (d_pred == d_gt)
            self.stick_dir_correct[i].update(dc.float().sum().item(), dc.numel())

            # 死区准确率
            in_dz_gt = (gt[valid] >= self.stick_center - self.stick_dz_half) & \
                       (gt[valid] <= self.stick_center + self.stick_dz_half)
            if in_dz_gt.any():
                in_dz_pred = (pred[valid][in_dz_gt] >= self.stick_center - self.stick_dz_half) & \
                             (pred[valid][in_dz_gt] <= self.stick_center + self.stick_dz_half)
                self.stick_dz_correct[i].update(in_dz_pred.float().sum().item(), in_dz_pred.numel())

        # ---- 扳机 (2 轴) ----
        trigger_logits = [
            action_logits.left_trigger,   # [B, T, 1, n_trigger_bins]
            action_logits.right_trigger,
        ]
        for i, logit in enumerate(trigger_logits):
            col = k + 4 + i
            pred = logit.argmax(dim=-1).squeeze(-1)  # [B, T]
            gt = masked_labels[:, :, col]
            valid = gt != -100
            if not valid.any():
                continue
            p, g = pred[valid].float(), gt[valid].float()
            ae = (p - g).abs()
            self.trigger_mae[i].update(ae.sum().item(), ae.numel())

            # 二分类: bin > 0 = 按下
            bp = (pred[valid] > 0)
            bg = (gt[valid] > 0)
            bc = (bp == bg)
            self.trigger_binary_correct[i].update(bc.float().sum().item(), bc.numel())

    def compute(self) -> Dict[str, Any]:
        """汇总所有指标，返回可序列化的 dict"""
        import math

        result: Dict[str, Any] = {}

        # ---- 按钮 ----
        result["button_accuracy"] = self.button_correct.mean
        result["button_exact_match"] = self.button_exact_match.mean
        result["button_sample_count"] = self.button_correct.count

        # macro F1
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

        # ---- 摇杆 ----
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

        # ---- 扳机 ----
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


def move_action_label_video_dataset_item_to_device(
    batch, device: torch.device, dtype: torch.dtype = None
):
    """Moves the tensors within an ActionLabelVideoDatasetItem to the specified device and dtype."""
    moved_frames = batch.frames.to(device)

    if isinstance(batch.action_annotations, torch.Tensor):
        moved_action_annotations = batch.action_annotations.to(device)
    else:
        moved_action_annotations = StructuredAction(
            keys=batch.action_annotations.keys.to(device),
            mouse_buttons=batch.action_annotations.mouse_buttons.to(device),
            mouse_delta_x=batch.action_annotations.mouse_delta_x.to(device),
            mouse_delta_y=batch.action_annotations.mouse_delta_y.to(device),
        )
    moved_env_subenv_encoding = batch.env_subenv_encoding.to(device)
    moved_user_action_mask = batch.user_action_mask.to(device)
    moved_system_action_mask = batch.system_action_mask.to(device)
    moved_valid_frame_mask = batch.valid_frame_mask.to(device)
    moved_text_embeddings = batch.text_embeddings.to(device)
    return ActionLabelVideoDatasetItem(
        frames=moved_frames,
        action_annotations=moved_action_annotations,
        env_subenv_encoding=moved_env_subenv_encoding,
        user_action_mask=moved_user_action_mask,
        system_action_mask=moved_system_action_mask,
        valid_frame_mask=moved_valid_frame_mask,
        text_embeddings=moved_text_embeddings,
    )


def find_all_checkpoints(path: str):
    if path.endswith(".ckpt"):
        return [path] if os.path.exists(path) else []

    if not os.path.isdir(path):
        return []

    ckpts = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".ckpt")]
    ckpts.sort(key=extract_step_from_checkpoint_path)
    return ckpts


def extract_step_from_checkpoint_path(checkpoint_path):
    match = re.search(r"step=(\d+)", checkpoint_path)

    if match:
        # group(1) contains the captured digits. int() handles leading zeros.
        global_step = int(match.group(1))
        print(f"Successfully extracted global_step: {global_step}")
    else:
        raise ValueError(
            f"Could not find global_step in the checkpoint path: {checkpoint_path}"
        )
    return global_step


def set_validation_dataset_cfg_to_single_thread(validation_dataset_cfgs, batch_size):
    # low numbers to not run out of space since the machine we run validation on is small and has small storage
    validation_dataset_cfgs.n_preprocess_threads_per_gpu = 2
    validation_dataset_cfgs.preprocessed_chunks_queue_size_per_gpu = 1
    validation_dataset_cfgs.dataset_worker_prefetch_factor = 2
    validation_dataset_cfgs.batch_size = batch_size
    return validation_dataset_cfgs


def _is_future_vision_model(config: LightningPolicyConfig) -> bool:
    return config.stage3_finetune.state_target_tokenizer is not None


def _load_validation_model(checkpoint_path: str, config: LightningPolicyConfig):
    if _is_future_vision_model(config):
        logging.info("Loading Stage3FutureVisionLightning for validation")
        model = Stage3FutureVisionLightning(
            config=config,
            inference_mode=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible_keys = model.load_state_dict(
            checkpoint["state_dict"], strict=False
        )

        missing_keys = list(incompatible_keys.missing_keys)
        unexpected_keys = list(incompatible_keys.unexpected_keys)
        allowed_missing_prefixes = ("state_target_tokenizer.",)
        disallowed_missing_keys = [
            key
            for key in missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]

        if missing_keys:
            logging.warning("Missing keys while loading future vision model: %s", missing_keys)
        if unexpected_keys:
            logging.warning(
                "Unexpected keys while loading future vision model: %s",
                unexpected_keys,
            )

        if disallowed_missing_keys or unexpected_keys:
            raise RuntimeError(
                "Unexpected checkpoint incompatibility for Stage3FutureVisionLightning. "
                f"disallowed_missing_keys={disallowed_missing_keys}, "
                f"unexpected_keys={unexpected_keys}"
            )
        return model

    logging.info("Loading Stage3LabelledBCLightning for validation")
    return Stage3LabelledBCLightning.load_from_checkpoint(
        checkpoint_path, config=config, inference_mode=True
    )


def _calculate_validation_loss(model, batch_to_cuda, actions_in, masked_labels):
    """返回 (loss, losses, auxiliary_outputs, action_logits_or_None)"""
    if isinstance(model, Stage3FutureVisionLightning):
        frames = model._normalize_frames(batch_to_cuda.frames)
        action_embeddings_in = model.action_in_to_tokens(actions_in)
        action_out_embeddings, _, future_vision_pred, auxiliary_losses, auxiliary_outputs = (
            model.transformer_forward_function(
                frames, action_embeddings_in, batch_to_cuda.text_embeddings
            )
        )
        future_vision_target = model._build_state_target(
            frames=frames,
            future_vision_pred=future_vision_pred,
        )
        future_vision_loss = torch.nn.functional.mse_loss(
            future_vision_pred, future_vision_target
        )
        action_logits = model.action_out_tokens_to_logits(action_out_embeddings)
        action_loss, losses = model._calculate_action_loss_from_logits(
            action_logits, masked_labels, auxiliary_losses
        )
        loss = action_loss + model.future_vision_loss_weight * future_vision_loss
        losses["future_vision"] = future_vision_loss
        return loss, losses, auxiliary_outputs, action_logits

    loss, _, losses, auxiliary_outputs = model._calculate_loss(
        batch_to_cuda,
        actions_in,
        masked_labels,
        batch_to_cuda.text_embeddings,
    )
    return loss, losses, auxiliary_outputs, None


def _flatten_gamepad_metrics(gp: Dict[str, Any], prefix: str) -> Dict[str, float]:
    """将嵌套的 gamepad metrics dict 展平为 wandb 可上报的 flat dict"""
    flat = {}
    skip_keys = {"per_button_f1", "per_stick", "per_trigger", "button_sample_count"}
    for k, v in gp.items():
        if k in skip_keys:
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_v is not None:
                        flat[f"{prefix}/{sub_k}"] = sub_v
            continue
        if v is not None:
            flat[f"{prefix}/{k}"] = v
    return flat


def report_validation_metrics(
    checkpoint_path: str, config_path: str, global_step: int, run_id: str
):
    BATCH_SIZE_FOR_VAL = 1
    t0 = time.time()

    try:
        config = load_config(config_path, LightningPolicyConfig)
        stage = "stage3_finetune"
        print(
            f"validating stage {stage}, run id {run_id}, global step {global_step}, "
            f"checkpoint {checkpoint_path}"
        )
        wandb_kwargs = dict(
            project=config.wandb.project,
            group=config.wandb.exp_name,
            name=config.wandb.exp_name + "_validation",
            job_type="validation",
        )
        # 如果配置中指定了entity，使用配置的；否则使用默认（用户登录的组织）
        if config.wandb.entity is not None:
            wandb_kwargs["entity"] = config.wandb.entity
        wandb_kwargs.update(dict(id=run_id, resume="allow"))
        wandb.init(**wandb_kwargs)
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step")
        # prepare dataset
        datamodule = Stage3DataModule(config)
        for idx in range(len(config.stage3_finetune.validation_datasets)):
            config.stage3_finetune.validation_datasets[idx] = (
                set_validation_dataset_cfg_to_single_thread(
                    config.stage3_finetune.validation_datasets[idx],
                    BATCH_SIZE_FOR_VAL,
                )
            )
        model = _load_validation_model(checkpoint_path, config)
        batch_size = max(
            [v.batch_size for v in config.stage3_finetune.validation_datasets],
            default=1,
        )
        n_validation_steps = config.stage3_finetune.n_validation_steps

        # Keep comparable sample count when forcing validation batch size to 1.
        n_validation_steps = max(
            1, n_validation_steps * batch_size // BATCH_SIZE_FOR_VAL
        )

        datamodule.setup(stage)
        val_dataloaders = datamodule.val_dataloader()
        del datamodule

        model.eval()
        model = model.to("cuda")

        # Ensure all block_masks are moved to the correct device
        # This is needed because block_masks might be created on CPU during model initialization
        if hasattr(model, "bc_transformer") and hasattr(
            model.bc_transformer, "block_mask_to_device"
        ):
            model.bc_transformer.block_mask_to_device(model.device)

        print(f"takes {time.time() - t0:.3f}s to prepare the dataset and model")
        # Get the model's dtype for input tensor conversion
        validation_metrics = {}
        model_dtype = next(model.parameters()).dtype

        # 判断是否为手柄模式
        is_gamepad = hasattr(model, '_use_gamepad_mapping') and model._use_gamepad_mapping()

        for val_set_name in val_dataloaders.keys():
            metrics = {
                "off_perplexity": LossMetric().to(model.device),
            }
            validation_metrics[val_set_name] = metrics

        # 为每个 val set 创建 GamepadActionMetrics（如果是手柄模式）
        gamepad_metrics_per_set: Dict[str, GamepadActionMetrics] = {}
        if is_gamepad:
            for val_set_name in val_dataloaders.keys():
                gamepad_metrics_per_set[val_set_name] = GamepadActionMetrics(
                    n_buttons=model.n_gamepad_button_actions,
                    n_stick_bins=model.n_gamepad_stick_bins,
                    n_trigger_bins=model.n_gamepad_trigger_bins,
                )

        start_time = time.time()
        for val_set_name, val_dataloader in val_dataloaders.items():
            print(f"\nProcessing validation set: {val_set_name}")
            val_metrics = validation_metrics[val_set_name]
            gp_metrics = gamepad_metrics_per_set.get(val_set_name)

            for batch_idx, batch in enumerate(val_dataloader):
                batch_to_cuda = move_action_label_video_dataset_item_to_device(
                    batch, "cuda", dtype=model_dtype
                )

                actions_in, masked_labels, _ = model._create_target_and_masked_labels(
                    batch_to_cuda
                )

                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    loss, losses, auxiliary_outputs, action_logits = _calculate_validation_loss(
                        model,
                        batch_to_cuda,
                        actions_in,
                        masked_labels,
                    )

                if torch.isnan(loss):
                    # When all labels are masked out for a batch, some losses may become NaN.
                    # Skip those batches to keep the whole validation run stable.
                    logging.warning(
                        "NaN loss at batch %s in %s, skipping this batch",
                        batch_idx,
                        val_set_name,
                    )
                    continue

                val_metrics["off_perplexity"].update(cross_entropy_to_perplexity(loss))
                for k, v in losses.items():
                    if k == "rz_loss" or k == "lb_loss":
                        continue
                    metric_name = f"off_perplexity_{k}"
                    if metric_name not in val_metrics:
                        val_metrics[metric_name] = LossMetric().to(model.device)
                    val_metrics[metric_name].update(cross_entropy_to_perplexity(v).item())

                # 手柄指标累积（在 GPU 上计算）
                if gp_metrics is not None and action_logits is not None and isinstance(action_logits, GamepadActionLogits):
                    gp_metrics.update(
                        action_logits=action_logits,
                        masked_labels=masked_labels,
                        n_buttons=model.n_gamepad_button_actions,
                    )

                if batch_idx % 100 == 0:
                    total_time = time.time() - start_time
                    start_time = time.time()
                    print(f"Batch {batch_idx}: Total processing time {total_time:.3f}s")

                if (batch_idx + 1) >= n_validation_steps:
                    break

        # ---- 汇总 & 输出 ----
        all_results: Dict[str, Any] = {
            "checkpoint": checkpoint_path,
            "global_step": int(global_step),
            "config_path": config_path,
        }

        for val_set_name, val_metrics in validation_metrics.items():
            set_results: Dict[str, Any] = {"perplexity_metrics": {}}
            for metric_name, metric in val_metrics.items():
                value = metric.compute()
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                set_results["perplexity_metrics"][metric_name] = value
                print({f"{val_set_name}_validation_{metric_name}": value})

            # 手柄指标
            if val_set_name in gamepad_metrics_per_set:
                gp_result = gamepad_metrics_per_set[val_set_name].compute()
                set_results["gamepad_metrics"] = gp_result
                print(f"\n{'='*60}")
                print(f"[{val_set_name}] Gamepad Action Metrics:")
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

            all_results[val_set_name] = set_results

        total_time = time.time() - t0
        all_results["total_validation_time_sec"] = round(total_time, 3)
        print(f"Total validation time: {total_time:.3f}s")

        # ---- 保存本地 JSON ----
        output_dir = os.path.dirname(checkpoint_path)
        json_path = os.path.join(
            output_dir,
            f"validation_step{global_step:08d}.json",
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Validation results saved to: {json_path}")

        # ---- wandb 上报（原有 perplexity + 新增 gamepad 指标）----
        for val_set_name, val_metrics in validation_metrics.items():
            log_data = {}
            for metric_name, metric in val_metrics.items():
                value = metric.compute()
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                log_data[f"{val_set_name}_validation_{metric_name}"] = value

            # 上报手柄指标到 wandb
            if val_set_name in gamepad_metrics_per_set:
                gp_result = gamepad_metrics_per_set[val_set_name].compute()
                _flat_gamepad = _flatten_gamepad_metrics(gp_result, prefix=f"{val_set_name}_validation_gamepad")
                log_data.update(_flat_gamepad)

            log_data["trainer/global_step"] = int(global_step)
            wandb.log(log_data, step=int(global_step))
    finally:
        wandb.finish()
        # ZMQ目录保持在/tmp（Linux原生文件系统）
        shutil.rmtree("/tmp/elefant_zmq", ignore_errors=True)
        shutil.rmtree("/ephemeral/elefant_tmp_data", ignore_errors=True)
        shutil.rmtree("/tmp/elefant_data", ignore_errors=True)
        logging.info("Cleaned tmp dataset dirs")


def is_step_in_range(
    step: int, min_steps: Optional[int], max_steps: Optional[int]
) -> bool:
    if min_steps is not None and step < min_steps:
        return False
    if max_steps is not None and step > max_steps:
        return False
    return True


def run_validation(
    checkpoint_dir: str,
    config_path: str,
    min_steps: Optional[int],
    max_steps: Optional[int],
):
    """
    Local execution path: run once through checkpoints.
    """
    run_id = wandb.util.generate_id()
    ckpts = find_all_checkpoints(checkpoint_dir)
    if not ckpts:
        logging.info(f"No checkpoints found in {checkpoint_dir}")
    for checkpoint_path in ckpts:
        try:
            global_step = extract_step_from_checkpoint_path(checkpoint_path)
        except ValueError:
            logging.warning(f"Skipping checkpoint with unparseable step: {checkpoint_path}")
            continue

        if not is_step_in_range(global_step, min_steps, max_steps):
            continue

        report_validation_metrics(checkpoint_path, config_path, global_step, run_id)
    logging.info("Validation metrics logged successfully.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint dir (or single .ckpt) to validate",
    )
    parser.add_argument("--config_path", type=str, required=True, help="Config path")
    parser.add_argument(
        "--min_steps",
        type=int,
        default=None,
        help="Minimum global step (inclusive) of checkpoints to validate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum global step (inclusive) of checkpoints to validate",
    )
    args = parser.parse_args()

    # sanity check the range if both are provided
    if args.min_steps is not None and args.max_steps is not None:
        if args.min_steps > args.max_steps:
            parser.error("--min_steps must be <= --max_steps")

    run_validation(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config_path,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
