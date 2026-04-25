"""
Per-offset validation — direct action paradigm.

训练时每帧设置 F 个 ain tokens，每个 a_in^f 经过 decoder 生成对应
当前帧 + 未来 (F-1) 帧的动作预测。

本脚本的验证思路：
  对每一帧 t，同时提取所有 F 个 ain token 的 logits，
  a_in^f 对应 offset=f 的 GT（即 t+f 帧的动作），
  分别计算各 offset 的性能指标，从而分析模型在
  "预测当前帧"、"预测1帧后"、...、"预测17帧后" 上的能力衰减曲线。

用法:
  uv run elefant/policy_model/validation_per_offset_action_direct.py \\
    --checkpoint_dir output/policy_model/.../stage3_finetune \\
    --config_path config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml \\
    --data_folder cuphead_dataset_converted \\
    --n_validation_steps 200 \\
    [--min_steps 100000] [--max_steps 130000] [--no_wandb]
"""

import argparse
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import torch
import wandb

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig, ValidationDatasetConfig
from elefant.policy_model.stage3_finetune import (
    Stage3DataModule,
    Stage3FutureVisionLightning,
    Stage3LabelledBCLightning,
    GamepadDirectActionLogits,
)
from elefant.policy_model.validation_action_direct import (
    DirectActionMetrics,
    _flatten_direct_metrics,
    extract_step_from_checkpoint_path,
    find_all_checkpoints,
    is_step_in_range,
    move_batch_to_device,
    set_validation_dataset_cfg_to_single_thread,
)


# ---------------------------------------------------------------------------
# Per-offset metrics container
# ---------------------------------------------------------------------------

class PerOffsetDirectMetrics:
    """
    为每个 future offset (0, 1, ..., F-1) 各维护一个 DirectActionMetrics 实例。

    offset=0  → a_in^0 预测当前帧动作
    offset=f  → a_in^f 预测 t+f 帧动作

    调用方式:
        metrics = PerOffsetDirectMetrics(n_offsets=F, n_buttons=..., n_stick_bins=...)
        metrics.update(offset=f, action_logits=logits_f, masked_labels=labels_f)
        result = metrics.compute()   # Dict[str, Dict]  key = f"offset_{f:02d}"
    """

    def __init__(self, n_offsets: int, n_buttons: int, n_stick_bins: int):
        self.n_offsets = n_offsets
        self.per_offset: List[DirectActionMetrics] = [
            DirectActionMetrics(n_buttons=n_buttons, n_stick_bins=n_stick_bins)
            for _ in range(n_offsets)
        ]

    def update(
        self,
        offset: int,
        action_logits: GamepadDirectActionLogits,
        masked_labels: torch.Tensor,
    ):
        """
        offset:        int, 0 <= offset < n_offsets
        action_logits: GamepadDirectActionLogits  [B, T, ...]
        masked_labels: [B, T, n_buttons+4]  已按 offset 移位（_build_future_masked_labels）
        """
        assert 0 <= offset < self.n_offsets, f"offset {offset} out of range [0, {self.n_offsets})"
        self.per_offset[offset].update(action_logits, masked_labels)

    def compute(self) -> Dict[str, Any]:
        """
        返回 {
            "offset_00": { button_accuracy, ..., per_button_f1, per_stick },
            "offset_01": { ... },
            ...
            "summary": {
                "offsets": [0, 1, ...],
                "button_accuracy":     [val_0, val_1, ...],
                "button_exact_match":  [...],
                "button_f1_macro":     [...],
                "stick_mae":           [...],
                "stick_rmse":          [...],
                "stick_direction_accuracy": [...],
                "stick_deadzone_accuracy":  [...],
            }
        }
        """
        result: Dict[str, Any] = {}
        summary_keys = [
            "button_accuracy", "button_exact_match", "button_f1_macro",
            "stick_mae", "stick_rmse", "stick_direction_accuracy", "stick_deadzone_accuracy",
        ]
        summary: Dict[str, Any] = {"offsets": list(range(self.n_offsets))}
        for k in summary_keys:
            summary[k] = []

        for f in range(self.n_offsets):
            m = self.per_offset[f].compute()
            result[f"offset_{f:02d}"] = m
            for k in summary_keys:
                summary[k].append(m.get(k))

        result["summary"] = summary
        return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_validation_datasets(
    config: LightningPolicyConfig,
    data_folder: Optional[str],
) -> None:
    """
    若 config 中 validation_datasets 为空，从 training_dataset 构造 fallback。
    同时用 data_folder 覆盖 local_prefix（若提供）。
    """
    stage3 = config.stage3_finetune
    if len(stage3.validation_datasets) == 0:
        fallback = stage3.training_dataset.model_dump()
        fallback["validation_name"] = "validation_0"
        stage3.validation_datasets = [ValidationDatasetConfig(**fallback)]

    for i, ds in enumerate(stage3.validation_datasets):
        if data_folder:
            ds.local_prefix = data_folder
        if not ds.validation_name:
            ds.validation_name = f"validation_{i}"

def _load_model(checkpoint_path: str, config: LightningPolicyConfig):
    has_future = (
        config.stage3_finetune.state_target_tokenizer is not None
        or config.stage3_finetune.use_precomputed_vision_features
    )
    if has_future:
        logging.info("Loading Stage3FutureVisionLightning for per-offset validation")
        model = Stage3FutureVisionLightning(config=config, inference_mode=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
        if incompatible.missing_keys:
            logging.warning("Missing keys: %s", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            logging.warning("Unexpected keys: %s", incompatible.unexpected_keys)
        return model

    logging.info("Loading Stage3LabelledBCLightning for per-offset validation")
    return Stage3LabelledBCLightning.load_from_checkpoint(
        checkpoint_path, config=config, inference_mode=True
    )


# ---------------------------------------------------------------------------
# Forward pass — 提取所有 F 个 ain token 的 logits
# ---------------------------------------------------------------------------

def _forward_all_offsets(
    model,
    batch_cuda,
    actions_in: torch.Tensor,
    masked_labels: torch.Tensor,
    F: int,
) -> List[GamepadDirectActionLogits]:
    """
    一次 forward，返回长度为 F 的列表，第 f 个元素是 a_in^f 对应的 logits。

    action_out_embeddings shape: [B, T, F, D]  (gamepad_direct 路径)
    """
    frames = model._normalize_frames(batch_cuda.frames)
    action_embeddings_in = model.action_in_to_tokens(actions_in)

    # future_offsets=[0,1,...,F-1] 确保 transformer 输出全部 F 个 ain token
    future_offsets = list(range(F))

    if isinstance(model, Stage3FutureVisionLightning):
        action_out_embeddings, _, future_vision_pred, auxiliary_losses, _ = (
            model.transformer_forward_function(
                frames,
                action_embeddings_in,
                batch_cuda.text_embeddings,
                future_offsets=future_offsets,
                skip_action_decoder=True,
            )
        )
    else:
        action_out_embeddings, _, auxiliary_losses, _ = (
            model.transformer_forward_function(
                frames,
                action_embeddings_in,
                batch_cuda.text_embeddings,
                future_offsets=future_offsets,
                skip_action_decoder=True,
            )
        )

    # action_out_embeddings: [B, T, F, D]
    logits_per_offset: List[GamepadDirectActionLogits] = []
    for f in range(F):
        a_in_f = action_out_embeddings[:, :, f, :]          # [B, T, D]
        logits_f = model.action_out_tokens_to_logits(a_in_f)
        logits_per_offset.append(logits_f)

    return logits_per_offset


# ---------------------------------------------------------------------------
# wandb flat helper
# ---------------------------------------------------------------------------

def _flatten_per_offset_metrics(
    per_offset_result: Dict[str, Any],
    prefix: str,
    F: int,
) -> Dict[str, float]:
    """将 per-offset 结果展平为 wandb 可上报的 flat dict。"""
    flat: Dict[str, float] = {}
    summary_scalar_keys = [
        "button_accuracy", "button_exact_match", "button_f1_macro",
        "stick_mae", "stick_rmse", "stick_direction_accuracy", "stick_deadzone_accuracy",
    ]
    for f in range(F):
        key = f"offset_{f:02d}"
        m = per_offset_result.get(key, {})
        flat.update(_flatten_direct_metrics(m, prefix=f"{prefix}/{key}"))

    # summary 折线图：每个 scalar 指标按 offset 上报
    summary = per_offset_result.get("summary", {})
    for sk in summary_scalar_keys:
        vals = summary.get(sk, [])
        for f, v in enumerate(vals):
            if v is not None:
                flat[f"{prefix}/summary/{sk}/offset_{f:02d}"] = v

    return flat


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def report_per_offset_validation_metrics(
    checkpoint_path: str,
    config_path: str,
    global_step: int,
    run_id: str,
    use_wandb: bool = True,
    data_folder: Optional[str] = None,
    n_validation_steps: Optional[int] = None,
):
    BATCH_SIZE_FOR_VAL = 1
    t0 = time.time()

    try:
        config = load_config(config_path, LightningPolicyConfig)
        F = config.policy_model.n_future_action_tokens

        print(
            f"[per-offset] validating run_id={run_id}, step={global_step}, "
            f"F={F}, checkpoint={checkpoint_path}"
        )

        if F < 2:
            logging.warning(
                "n_future_action_tokens=%d — per-offset validation is only meaningful "
                "when F >= 2. Proceeding anyway (only offset_00 will be populated).",
                F,
            )

        if use_wandb:
            wandb_kwargs = dict(
                project=config.wandb.project,
                group=config.wandb.exp_name,
                name=config.wandb.exp_name + "_validation_per_offset",
                job_type="validation",
                id=run_id,
                resume="allow",
            )
            if config.wandb.entity is not None:
                wandb_kwargs["entity"] = config.wandb.entity
            wandb.init(**wandb_kwargs)
            wandb.define_metric("trainer/global_step")
            wandb.define_metric("*", step_metric="trainer/global_step")

        _ensure_validation_datasets(config, data_folder)

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
        # 命令行 --n_validation_steps 优先；否则从 config 换算
        if n_validation_steps is not None:
            effective_n_steps = max(1, n_validation_steps)
        else:
            cfg_steps = config.stage3_finetune.n_validation_steps
            effective_n_steps = max(1, cfg_steps * batch_size // BATCH_SIZE_FOR_VAL)

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
            "This validation script requires action_mapping_type='gamepad_direct'."
        )

        print(
            f"Model ready in {time.time() - t0:.1f}s  |  "
            f"n_future_action_tokens={F}  "
            f"n_direct_buttons={model.n_direct_buttons}  "
            f"n_direct_stick_bins={model.n_direct_stick_bins}"
        )

        # per-set accumulators
        per_offset_metrics: Dict[str, PerOffsetDirectMetrics] = {}
        for name in val_dataloaders:
            per_offset_metrics[name] = PerOffsetDirectMetrics(
                n_offsets=F,
                n_buttons=model.n_direct_buttons,
                n_stick_bins=model.n_direct_stick_bins,
            )

        start_time = time.time()
        for val_set_name, val_dataloader in val_dataloaders.items():
            print(f"\nProcessing validation set: {val_set_name}")
            po_m = per_offset_metrics[val_set_name]

            for batch_idx, batch in enumerate(val_dataloader):
                batch_cuda = move_batch_to_device(batch, "cuda")
                actions_in, masked_labels, _ = model._create_target_and_masked_labels(batch_cuda)

                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    logits_per_offset = _forward_all_offsets(
                        model, batch_cuda, actions_in, masked_labels, F
                    )

                # 对每个 offset 用移位后的 labels 更新指标
                for f, logits_f in enumerate(logits_per_offset):
                    labels_f = model._build_future_masked_labels(masked_labels, offset=f)
                    if isinstance(logits_f, GamepadDirectActionLogits):
                        po_m.update(offset=f, action_logits=logits_f, masked_labels=labels_f)

                if batch_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    start_time = time.time()
                    print(f"  batch {batch_idx}: {elapsed:.1f}s")

                if (batch_idx + 1) >= effective_n_steps:
                    break

        # ---- aggregate ----
        all_results: Dict[str, Any] = {
            "checkpoint": checkpoint_path,
            "global_step": int(global_step),
            "config_path": config_path,
            "n_future_action_tokens": F,
        }

        for val_set_name in val_dataloaders:
            po_m = per_offset_metrics[val_set_name]
            result = po_m.compute()
            all_results[val_set_name] = {"per_offset_metrics": result}

            print(f"\n{'='*60}")
            print(f"[{val_set_name}] Per-Offset Direct Action Metrics (F={F}):")
            print(f"{'='*60}")
            print(f"  {'offset':>8}  {'btn_acc':>8}  {'btn_em':>8}  {'btn_f1':>8}  {'stk_mae':>8}  {'stk_dir':>8}")
            print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
            for f in range(F):
                m = result.get(f"offset_{f:02d}", {})
                print(
                    f"  {f:>8d}  "
                    f"{_fmt(m.get('button_accuracy')):>8}  "
                    f"{_fmt(m.get('button_exact_match')):>8}  "
                    f"{_fmt(m.get('button_f1_macro')):>8}  "
                    f"{_fmt(m.get('stick_mae')):>8}  "
                    f"{_fmt(m.get('stick_direction_accuracy')):>8}"
                )
            print(f"{'='*60}")

        total_time = time.time() - t0
        all_results["total_validation_time_sec"] = round(total_time, 3)
        print(f"Total validation time: {total_time:.1f}s")

        # ---- save JSON ----
        output_dir = os.path.dirname(checkpoint_path)
        json_path = os.path.join(
            output_dir, f"validation_per_offset_step{global_step:08d}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")

        # ---- wandb ----
        if use_wandb:
            for val_set_name in val_dataloaders:
                result = all_results[val_set_name]["per_offset_metrics"]
                log_data = _flatten_per_offset_metrics(
                    result,
                    prefix=f"{val_set_name}_validation_per_offset",
                    F=F,
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


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "  None"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_validation(
    checkpoint_dir: str,
    config_path: str,
    min_steps: Optional[int],
    max_steps: Optional[int],
    use_wandb: bool = True,
    data_folder: Optional[str] = None,
    n_validation_steps: Optional[int] = None,
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
        report_per_offset_validation_metrics(
            checkpoint_path, config_path, global_step, run_id, use_wandb,
            data_folder=data_folder,
            n_validation_steps=n_validation_steps,
        )
    logging.info("Validation complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Per-offset validation — direct action paradigm (F future tokens)"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00130000.ckpt")
    parser.add_argument("--config_path", type=str, default="config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml")
    parser.add_argument("--min_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--data_folder", type=str, default=None,
                        help="覆盖 config 中的 local_prefix（数据集根目录），"
                             "config 无 validation_datasets 时必须指定")
    parser.add_argument("--n_validation_steps", type=int, default=None,
                        help="评估的 batch 数量，覆盖 config 中的 n_validation_steps")
    parser.add_argument("--no_wandb", action="store_true",
                        help="禁用 wandb 上报，仅保存本地 JSON")
    args = parser.parse_args()
    print(f"args: {args}")
    if args.min_steps is not None and args.max_steps is not None:
        if args.min_steps > args.max_steps:
            parser.error("--min_steps must be <= --max_steps")

    run_validation(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config_path,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        use_wandb=not args.no_wandb,
        data_folder=args.data_folder,
        n_validation_steps=args.n_validation_steps,
    )


if __name__ == "__main__":
    main()

