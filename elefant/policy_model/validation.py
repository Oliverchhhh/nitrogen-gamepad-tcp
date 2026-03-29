"""
Example:
uv run elefant/policy_model/validation.py \
  --checkpoint_dir=output/policy_model/150M_nitrogen/stage3_finetune \
  --config_path=config/policy_model/150M_local_nitrogen_dataset_current.yaml
"""

import re
import argparse
import os
import wandb
import logging
import shutil
import torch
import time

from typing import Optional

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.stage3_finetune import (
    Stage3DataModule,
    Stage3LabelledBCLightning,
    Stage3FutureVisionLightning,
)
from elefant.data import ActionLabelVideoDatasetItem, StructuredAction
from elefant.metrics import LossMetric
from elefant.torch import cross_entropy_to_perplexity


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
        return loss, losses, auxiliary_outputs

    loss, _, losses, auxiliary_outputs = model._calculate_loss(
        batch_to_cuda,
        actions_in,
        masked_labels,
        batch_to_cuda.text_embeddings,
    )
    return loss, losses, auxiliary_outputs


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
        for val_set_name in val_dataloaders.keys():
            metrics = {
                "off_perplexity": LossMetric().to(model.device),
            }
            validation_metrics[val_set_name] = metrics

        start_time = time.time()
        for val_set_name, val_dataloader in val_dataloaders.items():
            print(f"\nProcessing validation set: {val_set_name}")
            val_metrics = validation_metrics[val_set_name]
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
                    loss, losses, auxiliary_outputs = _calculate_validation_loss(
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

                if batch_idx % 100 == 0:
                    total_time = time.time() - start_time
                    start_time = time.time()
                    print(f"Batch {batch_idx}: Total processing time {total_time:.3f}s")

                if (batch_idx + 1) >= n_validation_steps:
                    break

        for val_set_name, val_metrics in validation_metrics.items():
            for metric_name, metric in val_metrics.items():
                value = metric.compute()
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                print({f"{val_set_name}_validation_{metric_name}": value})

        total_time = time.time() - t0
        print(f"Total validation time: {total_time:.3f}s")

        for val_set_name, val_metrics in validation_metrics.items():
            log_data = {}
            for metric_name, metric in val_metrics.items():
                value = metric.compute()
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                log_data[f"{val_set_name}_validation_{metric_name}"] = value
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
