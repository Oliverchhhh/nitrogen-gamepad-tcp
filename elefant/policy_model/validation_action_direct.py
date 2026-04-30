"""
Validation script for the action-direct (non-autoregressive) paradigm.

Example:
uv run elefant/policy_model/validation_action_direct.py \
  --checkpoint_dir=output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head/stage3_finetune \
  --config_path=config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml
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

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch.nn.functional as F

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.stage3_finetune import (
    Stage3DataModule,
    Stage3FutureVisionLightning,
    Stage3LabelledBCLightning,
    GamepadDirectActionLogits,
)
from elefant.data import ActionLabelVideoDatasetItem, StructuredAction
from elefant.metrics import LossMetric
from elefant.torch import cross_entropy_to_perplexity


# ---------------------------------------------------------------------------
# Running mean accumulator
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


# ---------------------------------------------------------------------------
# Direct-action metrics  (button BCE + stick CE, no trigger)
# ---------------------------------------------------------------------------

class DirectActionMetrics:
    """
    Metrics for the direct (non-autoregressive) gamepad head.

    Buttons  — binary per-button:
        button_accuracy      per-button frame-level accuracy  (sigmoid > 0.5)
        button_exact_match   all buttons correct in one frame
        button_f1_macro      macro F1 across buttons

    Sticks   — 4-axis classification (lx, ly, rx, ry):
        stick_mae            mean absolute bin error (4-axis avg)
        stick_rmse           root mean squared bin error
        stick_direction_acc  3-class direction accuracy (left/neutral/right)
        stick_deadzone_acc   GT-in-deadzone → pred-in-deadzone rate
    """

    STICK_AXES = ["left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"]

    def __init__(self, n_buttons: int, n_stick_bins: int):
        self.n_buttons = n_buttons
        self.n_stick_bins = n_stick_bins
        self.stick_center = n_stick_bins // 2
        # For 3-bin sticks the center bin is the only deadzone bin.
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
    def _to_direction(bin_idx: torch.Tensor, center: int, dz: int) -> torch.Tensor:
        d = torch.zeros_like(bin_idx)
        d[bin_idx < center - dz] = -1
        d[bin_idx > center + dz] = 1
        return d

    def update(
        self,
        action_logits: GamepadDirectActionLogits,
        masked_labels: torch.Tensor,
    ):
        """
        Args:
            action_logits: GamepadDirectActionLogits
                .buttons       [B, T, n_buttons]   raw logits (BCEWithLogits)
                .left_stick_x  [B, T, n_stick_bins]
                .left_stick_y  [B, T, n_stick_bins]
                .right_stick_x [B, T, n_stick_bins]
                .right_stick_y [B, T, n_stick_bins]
            masked_labels: [B, T, n_buttons + 4]   (ignore_index = -100)
        """
        k = self.n_buttons

        # ---- buttons ----
        pred_btn = (action_logits.buttons.sigmoid() > 0.5).long()  # [B, T, k]
        gt_btn = masked_labels[:, :, :k]                            # [B, T, k]
        valid_btn = gt_btn != -100

        if valid_btn.any():
            correct = pred_btn[valid_btn] == gt_btn[valid_btn]
            self.btn_correct.update(correct.float().sum().item(), correct.numel())

            frame_valid = valid_btn.all(dim=-1)  # [B, T]
            if frame_valid.any():
                frame_correct = (pred_btn == gt_btn).all(dim=-1)
                em = frame_correct[frame_valid]
                self.btn_exact_match.update(em.float().sum().item(), em.numel())

            for bi in range(self.n_buttons):
                v = valid_btn[:, :, bi]
                if not v.any():
                    continue
                p = pred_btn[:, :, bi][v] > 0
                g = gt_btn[:, :, bi][v] > 0
                self._btn_tp[bi] += (p & g).sum().item()
                self._btn_fp[bi] += (p & ~g).sum().item()
                self._btn_fn[bi] += (~p & g).sum().item()

        # ---- sticks ----
        stick_logit_list = [
            action_logits.left_stick_x,
            action_logits.left_stick_y,
            action_logits.right_stick_x,
            action_logits.right_stick_y,
        ]
        for i, logit in enumerate(stick_logit_list):
            col = k + i
            pred = logit.argmax(dim=-1)          # [B, T]
            gt = masked_labels[:, :, col]        # [B, T]
            valid = gt != -100
            if not valid.any():
                continue
            p, g = pred[valid].float(), gt[valid].float()
            ae = (p - g).abs()
            self.stick_mae[i].update(ae.sum().item(), ae.numel())
            self.stick_sq_err[i].update((ae ** 2).sum().item(), ae.numel())

            d_pred = self._to_direction(pred[valid], self.stick_center, self.stick_dz_half)
            d_gt = self._to_direction(gt[valid], self.stick_center, self.stick_dz_half)
            dc = d_pred == d_gt
            self.stick_dir_correct[i].update(dc.float().sum().item(), dc.numel())

            in_dz_gt = (
                (gt[valid] >= self.stick_center - self.stick_dz_half)
                & (gt[valid] <= self.stick_center + self.stick_dz_half)
            )
            if in_dz_gt.any():
                in_dz_pred = (
                    (pred[valid][in_dz_gt] >= self.stick_center - self.stick_dz_half)
                    & (pred[valid][in_dz_gt] <= self.stick_center + self.stick_dz_half)
                )
                self.stick_dz_correct[i].update(
                    in_dz_pred.float().sum().item(), in_dz_pred.numel()
                )

    def compute(self) -> Dict[str, Any]:
        import math

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
# Helpers shared with validation.py
# ---------------------------------------------------------------------------

def move_batch_to_device(batch, device: torch.device):
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
    precomputed = getattr(batch, "precomputed_vision_features", None)
    return ActionLabelVideoDatasetItem(
        frames=moved_frames,
        action_annotations=moved_action_annotations,
        env_subenv_encoding=batch.env_subenv_encoding.to(device),
        user_action_mask=batch.user_action_mask.to(device), #[B, T] #全0
        system_action_mask=batch.system_action_mask.to(device), #[B, T] #全0
        valid_frame_mask=batch.valid_frame_mask.to(device), #[B, T] #全1
        text_embeddings=batch.text_embeddings.to(device), # [B, T, 1, D_text] #全0
        precomputed_vision_features=precomputed.to(device) if precomputed is not None else None,
    )


def find_all_checkpoints(path: str):
    if path.endswith(".ckpt"):
        return [path] if os.path.exists(path) else []
    if not os.path.isdir(path):
        return []
    ckpts = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".ckpt")]
    ckpts.sort(key=extract_step_from_checkpoint_path)
    return ckpts


def extract_step_from_checkpoint_path(checkpoint_path: str) -> int:
    match = re.search(r"step=(\d+)", checkpoint_path)
    if not match:
        raise ValueError(f"Could not find global_step in checkpoint path: {checkpoint_path}")
    step = int(match.group(1))
    print(f"Successfully extracted global_step: {step}")
    return step


def set_validation_dataset_cfg_to_single_thread(cfg, batch_size: int):
    cfg.n_preprocess_threads_per_gpu = 2
    cfg.preprocessed_chunks_queue_size_per_gpu = 1
    cfg.dataset_worker_prefetch_factor = 2
    cfg.batch_size = batch_size
    cfg.shuffle = False  # 关闭 shuffle，保证序列顺序确定性（TF/AR 对比时必须一致）
    return cfg


def is_step_in_range(step: int, min_steps: Optional[int], max_steps: Optional[int]) -> bool:
    if min_steps is not None and step < min_steps:
        return False
    if max_steps is not None and step > max_steps:
        return False
    return True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, config: LightningPolicyConfig):
    """Load Stage3FutureVisionLightning or Stage3LabelledBCLightning."""
    has_future = (
        config.stage3_finetune.state_target_tokenizer is not None
        or config.stage3_finetune.use_precomputed_vision_features
    )
    if has_future:
        logging.info("Loading Stage3FutureVisionLightning for direct-action validation")
        model = Stage3FutureVisionLightning(config=config, inference_mode=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
        if incompatible.missing_keys:
            logging.warning("Missing keys: %s", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            logging.warning("Unexpected keys: %s", incompatible.unexpected_keys)
        return model

    logging.info("Loading Stage3LabelledBCLightning for direct-action validation")
    return Stage3LabelledBCLightning.load_from_checkpoint(
        checkpoint_path, config=config, inference_mode=True
    )


# ---------------------------------------------------------------------------
# Forward pass — direct action paradigm
# ---------------------------------------------------------------------------

def _select_direct_frame_logits(
    action_logits_all: GamepadDirectActionLogits, frame_idx: int
) -> GamepadDirectActionLogits:
    return GamepadDirectActionLogits(
        buttons=action_logits_all.buttons[:, :, frame_idx, :],
        left_stick_x=action_logits_all.left_stick_x[:, :, frame_idx, :],
        left_stick_y=action_logits_all.left_stick_y[:, :, frame_idx, :],
        right_stick_x=action_logits_all.right_stick_x[:, :, frame_idx, :],
        right_stick_y=action_logits_all.right_stick_y[:, :, frame_idx, :],
    )


def _average_loss_dict(loss_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not loss_dicts:
        return {}
    out: Dict[str, torch.Tensor] = {}
    for d in loss_dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    n = len(loss_dicts)
    for k in list(out.keys()):
        out[k] = out[k] / n
    return out


def _forward_direct(
    model,
    batch_to_cuda,
    actions_in,
    masked_labels,
    frame_index: int = 0,
    all_future_frames: bool = False,
):
    """
    Run one forward pass.

    Works for both Stage3LabelledBCLightning and Stage3FutureVisionLightning
    when the model uses gamepad_direct_action_mapping.

    Returns:
        loss: scalar
        losses: dict
        action_logits: GamepadDirectActionLogits for single-frame mode, else None
        frame_pairs: list[(frame_idx, logits, shifted_labels)] for all-frame mode
        n_future_frames: int
    """
    frames = model._normalize_frames(batch_to_cuda.frames)
    B = frames.shape[0]
    T = frames.shape[1]
    action_embeddings_in = model.action_in_to_tokens(actions_in)

    if isinstance(model, Stage3FutureVisionLightning):
        action_out_embeddings, _, future_vision_pred, auxiliary_losses, _ = (
            model.transformer_forward_function(
                frames,
                action_embeddings_in,
                batch_to_cuda.text_embeddings,
                skip_action_decoder=True,
            )
        )
        # a_in^0 → [B, T, F, ...] logits
        action_logits_all = model.action_out_tokens_to_logits(action_out_embeddings[:, :, 0, :])
        n_future_frames = action_logits_all.buttons.shape[2]

        if all_future_frames:
            frame_pairs: List[Tuple[int, GamepadDirectActionLogits, torch.Tensor]] = []
            action_losses: List[torch.Tensor] = []
            loss_dicts: List[Dict[str, torch.Tensor]] = []
            for i in range(n_future_frames):
                labels_i = model._build_future_masked_labels(masked_labels, i)
                action_logits_i = _select_direct_frame_logits(action_logits_all, i)
                aux_i = auxiliary_losses if i == 0 else {}
                action_loss_i, _, losses_i = model._calculate_action_losses_from_logits_eager(
                    action_logits_i, labels_i, aux_i, B, T
                )
                frame_pairs.append((i, action_logits_i, labels_i))
                action_losses.append(action_loss_i)
                loss_dicts.append(losses_i)
            action_loss = sum(action_losses) / len(action_losses)
            losses = _average_loss_dict(loss_dicts)
            action_logits = None
        else:
            if frame_index < 0 or frame_index >= n_future_frames:
                raise ValueError(
                    f"frame_index={frame_index} out of range [0, {n_future_frames-1}]"
                )
            labels_eval = (
                model._build_future_masked_labels(masked_labels, frame_index)
                if frame_index > 0
                else masked_labels
            )
            action_logits = _select_direct_frame_logits(action_logits_all, frame_index)
            action_loss, _, losses = model._calculate_action_losses_from_logits_eager(
                action_logits, labels_eval, auxiliary_losses, B, T
            )
            frame_pairs = []

        future_vision_target = model._build_state_target(
            frames=frames,
            future_vision_pred=future_vision_pred,
            precomputed_vision_features=getattr(batch_to_cuda, "precomputed_vision_features", None),
        )
        future_vision_loss = F.mse_loss(future_vision_pred, future_vision_target)
        loss = action_loss + model.future_vision_loss_weight * future_vision_loss
        losses["future_vision"] = future_vision_loss
    else:
        action_out_embeddings, _, auxiliary_losses, _ = (
            model.transformer_forward_function(
                frames,
                action_embeddings_in,
                batch_to_cuda.text_embeddings,
                skip_action_decoder=True,
            )
        )
        # a_in^0 → [B, T, F, ...] logits
        action_logits_all = model.action_out_tokens_to_logits(action_out_embeddings[:, :, 0, :])
        n_future_frames = action_logits_all.buttons.shape[2]

        if all_future_frames:
            frame_pairs = []
            all_losses = []
            loss_dicts = []
            for i in range(n_future_frames):
                labels_i = model._build_future_masked_labels(masked_labels, i)
                action_logits_i = _select_direct_frame_logits(action_logits_all, i)
                aux_i = auxiliary_losses if i == 0 else {}
                loss_i, _, losses_i = model._calculate_action_losses_from_logits_eager(
                    action_logits_i, labels_i, aux_i, B, T
                )
                frame_pairs.append((i, action_logits_i, labels_i))
                all_losses.append(loss_i)
                loss_dicts.append(losses_i)
            loss = sum(all_losses) / len(all_losses)
            losses = _average_loss_dict(loss_dicts)
            action_logits = None
        else:
            if frame_index < 0 or frame_index >= n_future_frames:
                raise ValueError(
                    f"frame_index={frame_index} out of range [0, {n_future_frames-1}]"
                )
            labels_eval = (
                model._build_future_masked_labels(masked_labels, frame_index)
                if frame_index > 0
                else masked_labels
            )
            action_logits = _select_direct_frame_logits(action_logits_all, frame_index)
            loss, _, losses = model._calculate_action_losses_from_logits_eager(
                action_logits, labels_eval, auxiliary_losses, B, T
            )
            frame_pairs = []

    return loss, losses, action_logits, frame_pairs, n_future_frames


# ---------------------------------------------------------------------------
# wandb flat helper
# ---------------------------------------------------------------------------

def _flatten_direct_metrics(d: Dict[str, Any], prefix: str) -> Dict[str, float]:
    skip = {"per_button_f1", "per_stick", "button_sample_count"}
    flat = {}
    for k, v in d.items():
        if k in skip:
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_v is not None:
                        flat[f"{prefix}/{sub_k}"] = sub_v
            continue
        if v is not None:
            flat[f"{prefix}/{k}"] = v
    return flat


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def report_validation_metrics(
    checkpoint_path: str,
    config_path: str,
    global_step: int,
    run_id: str,
    use_wandb: bool = True,
    n_sequences: Optional[int] = None,
    frame_index: int = 0,
    all_future_frames: bool = False,
    data_folder: Optional[str] = None,
):
    BATCH_SIZE_FOR_VAL = 1
    t0 = time.time()

    try:
        config = load_config(config_path, LightningPolicyConfig)

        eval_mode = "all_future_frames" if all_future_frames else f"frame_index={frame_index}"
        print(
            f"[direct] validating run_id={run_id}, step={global_step}, "
            f"checkpoint={checkpoint_path}, mode={eval_mode}"
        )

        if use_wandb:
            wandb_kwargs = dict(
                project=config.wandb.project,
                group=config.wandb.exp_name,
                name=config.wandb.exp_name + "_validation",
                job_type="validation",
                id=run_id,
                resume="allow",
            )
            if config.wandb.entity is not None:
                wandb_kwargs["entity"] = config.wandb.entity
            wandb.init(**wandb_kwargs)
            wandb.define_metric("trainer/global_step")
            wandb.define_metric("*", step_metric="trainer/global_step")

        # dataset
        for idx in range(len(config.stage3_finetune.validation_datasets)):
            config.stage3_finetune.validation_datasets[idx] = (
                set_validation_dataset_cfg_to_single_thread(
                    config.stage3_finetune.validation_datasets[idx],
                    BATCH_SIZE_FOR_VAL,
                )
            )
        datamodule = Stage3DataModule(config)
        model = _load_model(checkpoint_path, config)

        batch_size = max(
            [v.batch_size for v in config.stage3_finetune.validation_datasets],
            default=1,
        )
        n_validation_steps = config.stage3_finetune.n_validation_steps
        n_validation_steps = max(1, n_validation_steps * batch_size // BATCH_SIZE_FOR_VAL)

        datamodule.setup("stage3_finetune")
        val_dataloaders = datamodule.val_dataloader()
        del datamodule

        model.eval()
        model = model.to("cuda")
        if hasattr(model, "bc_transformer") and hasattr(
            model.bc_transformer, "block_mask_to_device"
        ):
            model.bc_transformer.block_mask_to_device(model.device)

        assert model._use_gamepad_direct_mapping(), (
            "This validation script requires action_mapping_type='gamepad_direct'. "
            "Use validation.py for other mapping types."
        )

        print(
            f"Model ready in {time.time() - t0:.1f}s  |  "
            f"n_direct_buttons={model.n_direct_buttons}  "
            f"n_direct_stick_bins={model.n_direct_stick_bins}"
        )

        # per-set accumulators
        perplexity_metrics: Dict[str, Dict] = {}
        direct_metrics: Dict[str, DirectActionMetrics] = {}
        per_frame_direct_metrics: Dict[str, List[DirectActionMetrics]] = {}
        n_eval_future_frames = getattr(
            config.policy_model, "n_future_frames", config.policy_model.n_future_action_tokens
        )
        if all_future_frames:
            print(f"Evaluating all future frames: 0..{n_eval_future_frames-1}")
        for name in val_dataloaders:
            perplexity_metrics[name] = {"off_perplexity": LossMetric().to(model.device)}
            direct_metrics[name] = DirectActionMetrics(
                n_buttons=model.n_direct_buttons,
                n_stick_bins=model.n_direct_stick_bins,
            )
            per_frame_direct_metrics[name] = [
                DirectActionMetrics(
                    n_buttons=model.n_direct_buttons,
                    n_stick_bins=model.n_direct_stick_bins,
                )
                for _ in range(n_eval_future_frames)
            ]

        start_time = time.time()
        for val_set_name, val_dataloader in val_dataloaders.items():
            print(f"\nProcessing validation set: {val_set_name}")
            ppl_m = perplexity_metrics[val_set_name]
            dir_m = direct_metrics[val_set_name]

            seq_count = 0
            for batch_idx, batch in enumerate(val_dataloader):
                batch_cuda = move_batch_to_device(batch, "cuda")
                actions_in, masked_labels, _ = model._create_target_and_masked_labels(batch_cuda)

                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    loss, losses, action_logits, frame_pairs, n_future_frames = _forward_direct(
                        model,
                        batch_cuda,
                        actions_in,
                        masked_labels,
                        frame_index=frame_index,
                        all_future_frames=all_future_frames,
                    )

                if torch.isnan(loss):
                    logging.warning(
                        "NaN loss at batch %d in %s, skipping", batch_idx, val_set_name
                    )
                    continue

                ppl_m["off_perplexity"].update(cross_entropy_to_perplexity(loss))
                for k, v in losses.items():
                    if k in ("rz_loss", "lb_loss"):
                        continue
                    key = f"off_perplexity_{k}"
                    if key not in ppl_m:
                        ppl_m[key] = LossMetric().to(model.device)
                    ppl_m[key].update(cross_entropy_to_perplexity(v).item())

                if all_future_frames:
                    if n_future_frames != n_eval_future_frames:
                        raise ValueError(
                            f"Runtime n_future_frames={n_future_frames} != expected {n_eval_future_frames}"
                        )
                    for frame_i, logits_i, labels_i in frame_pairs:
                        dir_m.update(logits_i, labels_i)
                        per_frame_direct_metrics[val_set_name][frame_i].update(logits_i, labels_i)
                elif isinstance(action_logits, GamepadDirectActionLogits):
                    labels_eval = (
                        model._build_future_masked_labels(masked_labels, frame_index)
                        if frame_index > 0
                        else masked_labels
                    )
                    dir_m.update(action_logits, labels_eval)

                if batch_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    start_time = time.time()
                    print(f"  batch {batch_idx}: {elapsed:.1f}s")

                seq_count += batch_cuda.frames.shape[0]
                if n_sequences is not None and seq_count >= n_sequences:
                    print(f"  reached n_sequences={n_sequences}, stopping.")
                    break

                if (batch_idx + 1) >= n_validation_steps:
                    break

        # ---- aggregate ----
        all_results: Dict[str, Any] = {
            "checkpoint": checkpoint_path,
            "global_step": int(global_step),
            "config_path": config_path,
        }

        for val_set_name in val_dataloaders:
            ppl_m = perplexity_metrics[val_set_name]
            dir_m = direct_metrics[val_set_name]
            set_results: Dict[str, Any] = {"perplexity_metrics": {}}

            for metric_name, metric in ppl_m.items():
                value = metric.compute()
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                set_results["perplexity_metrics"][metric_name] = value
                print({f"{val_set_name}_validation_{metric_name}": value})

            dm = dir_m.compute()
            set_results["direct_action_metrics"] = dm
            if all_future_frames:
                per_frame = {}
                for frame_i in range(n_eval_future_frames):
                    per_frame[f"frame_{frame_i:02d}"] = (
                        per_frame_direct_metrics[val_set_name][frame_i].compute()
                    )
                set_results["direct_action_metrics_per_frame"] = per_frame
            print(f"\n{'='*60}")
            print(f"[{val_set_name}] Direct Action Metrics:")
            print(f"{'='*60}")
            print(f"  Button accuracy:        {dm.get('button_accuracy')}")
            print(f"  Button exact match:     {dm.get('button_exact_match')}")
            print(f"  Button F1 (macro):      {dm.get('button_f1_macro')}")
            print(f"  Stick MAE (bin):        {dm.get('stick_mae')}")
            print(f"  Stick RMSE (bin):       {dm.get('stick_rmse')}")
            print(f"  Stick direction acc:    {dm.get('stick_direction_accuracy')}")
            print(f"  Stick deadzone acc:     {dm.get('stick_deadzone_accuracy')}")
            print(f"{'='*60}")

            all_results[val_set_name] = set_results

        total_time = time.time() - t0
        all_results["total_validation_time_sec"] = round(total_time, 3)
        print(f"Total validation time: {total_time:.1f}s")

        # ---- save JSON ----
        output_dir = os.path.dirname(checkpoint_path)
        if data_folder:
            # 用数据集目录的最后一级名称作为后缀，去掉路径分隔符
            dataset_suffix = "_" + os.path.basename(data_folder.rstrip("/\\"))
        else:
            dataset_suffix = ""
        json_path = os.path.join(
            output_dir, f"validation_direct_step{global_step:08d}{dataset_suffix}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")

        # ---- wandb ----
        if use_wandb:
            for val_set_name in val_dataloaders:
                log_data: Dict[str, Any] = {}
                for metric_name, metric in perplexity_metrics[val_set_name].items():
                    value = metric.compute()
                    if isinstance(value, torch.Tensor):
                        value = value.detach().item()
                    log_data[f"{val_set_name}_validation_{metric_name}"] = value

                dm = direct_metrics[val_set_name].compute()
                log_data.update(
                    _flatten_direct_metrics(dm, prefix=f"{val_set_name}_validation_direct")
                )
                if all_future_frames:
                    for frame_i in range(n_eval_future_frames):
                        dmf = per_frame_direct_metrics[val_set_name][frame_i].compute()
                        log_data.update(
                            _flatten_direct_metrics(
                                dmf,
                                prefix=f"{val_set_name}_validation_direct/frame_{frame_i:02d}",
                            )
                        )
                log_data["trainer/global_step"] = int(global_step)
                wandb.log(log_data, step=int(global_step))

    finally:
        if use_wandb:
            wandb.finish()
        shutil.rmtree("/tmp/elefant_zmq", ignore_errors=True)
        shutil.rmtree("/ephemeral/elefant_tmp_data", ignore_errors=True)
        shutil.rmtree("/tmp/elefant_data", ignore_errors=True)
        logging.info("Cleaned tmp dataset dirs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_validation(
    checkpoint_dir: str,
    config_path: str,
    min_steps: Optional[int],
    max_steps: Optional[int],
    use_wandb: bool = True,
    n_sequences: Optional[int] = None,
    frame_index: int = 0,
    all_future_frames: bool = False,
    data_folder: Optional[str] = None,
):
    run_id = wandb.util.generate_id() if use_wandb else "local"
    ckpts = find_all_checkpoints(checkpoint_dir)
    if not ckpts:
        logging.info("No checkpoints found in %s", checkpoint_dir)
    for checkpoint_path in ckpts:
        try:
            global_step = extract_step_from_checkpoint_path(checkpoint_path)
        except ValueError:
            logging.warning("Skipping checkpoint with unparseable step: %s", checkpoint_path)
            continue
        if not is_step_in_range(global_step, min_steps, max_steps):
            continue
        report_validation_metrics(
            checkpoint_path,
            config_path,
            global_step,
            run_id,
            use_wandb,
            n_sequences=n_sequences,
            frame_index=frame_index,
            all_future_frames=all_future_frames,
            data_folder=data_folder,
        )
    logging.info("Validation complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--min_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="禁用 wandb 上报，仅保存本地 JSON")
    parser.add_argument("--n_sequences", type=int, default=None,
                        help="限制验证的轨迹序列数量，与 AR 脚本的 --n_sequences 语义对齐（默认跑完整个验证集）")
    parser.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="仅评估指定 future frame 索引（默认 0）。当 --all_future_frames 开启时忽略。",
    )
    parser.add_argument(
        "--all_future_frames",
        action="store_true",
        help="逐帧评估全部 future frames 并输出汇总与逐帧指标。",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=None,
        help="验证数据集路径，用于在输出 JSON 文件名中附加数据集名称后缀。",
    )
    args = parser.parse_args()
    print(f"args: {args}")
    if args.min_steps is not None and args.max_steps is not None:
        if args.min_steps > args.max_steps:
            parser.error("--min_steps must be <= --max_steps")
    if args.frame_index < 0:
        parser.error("--frame_index must be >= 0")

    run_validation(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config_path,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        use_wandb=not args.no_wandb,
        n_sequences=args.n_sequences,
        frame_index=args.frame_index,
        all_future_frames=args.all_future_frames,
        data_folder=args.data_folder,
    )


if __name__ == "__main__":
    main()
