"""
StaMo co-training ablation: how many future visual observations (s0 tokens)
should a0 attend at inference time?

Four conditions:
  attend_k=0  — a0 sees only img + itself  (co-training signal only, no prediction benefit)
  attend_k=1  — a0 sees 1 predicted s0
  attend_k=5  — a0 sees 5 predicted s0 tokens
  attend_k=10 — a0 sees all 10 s0 tokens  (same as training)

Usage:
    python scripts/validate_stamo_cotrain_ablation.py \\
        --config config/policy_model/150M_stamo_cotrain.yaml \\
        --checkpoint output/policy_model/150M_stamo_cotrain_N10/stage3_stamo_cotrain/step=20000.ckpt \\
        --data_folder /path/to/cuphead \\
        --n_batches 200 \\
        --batch_size 4 \\
        --attend_ks 0 1 5 10

Metrics reported per condition:
  button_accuracy, button_f1_macro
  stick_mae, stick_direction_accuracy
  action_loss (total CE/BCE across all 18 predicted frames)
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.stage3_finetune import (
    Stage3StaMoFutureVisionLightning,
    Stage3DataModule,
    GamepadDirectActionLogits,
)
from elefant.data import ActionLabelVideoDatasetItem, StructuredAction
from elefant.policy_model.validation_action_direct import (
    DirectActionMetrics,
    set_validation_dataset_cfg_to_single_thread,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_batch(batch, device):
    annotations = batch.action_annotations
    if not isinstance(annotations, torch.Tensor):
        annotations = StructuredAction(
            keys=annotations.keys.to(device),
            mouse_buttons=annotations.mouse_buttons.to(device),
            mouse_delta_x=annotations.mouse_delta_x.to(device),
            mouse_delta_y=annotations.mouse_delta_y.to(device),
        )
    precomputed_stamo = getattr(batch, "precomputed_stamo_features", None)
    return ActionLabelVideoDatasetItem(
        frames=batch.frames.to(device),
        action_annotations=annotations,
        env_subenv_encoding=batch.env_subenv_encoding.to(device),
        user_action_mask=batch.user_action_mask.to(device),
        system_action_mask=batch.system_action_mask.to(device),
        valid_frame_mask=batch.valid_frame_mask.to(device),
        text_embeddings=batch.text_embeddings.to(device),
        precomputed_vision_features=None,
        precomputed_stamo_features=(
            precomputed_stamo.to(device) if precomputed_stamo is not None else None
        ),
    )


def _load_model(checkpoint_path: str, config: LightningPolicyConfig) -> Stage3StaMoFutureVisionLightning:
    logging.info("Loading Stage3StaMoFutureVisionLightning from %s", checkpoint_path)
    model = Stage3StaMoFutureVisionLightning(config=config, inference_mode=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
    if incompatible.missing_keys:
        logging.warning("Missing keys: %s", incompatible.missing_keys[:10])
    if incompatible.unexpected_keys:
        logging.warning("Unexpected keys: %s", incompatible.unexpected_keys[:10])
    return model


# ---------------------------------------------------------------------------
# Single-condition evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_condition(
    model: Stage3StaMoFutureVisionLightning,
    dataloader,
    n_attend: int,
    n_batches: int,
    device: torch.device,
) -> dict:
    """Run eval loop for one n_future_attend condition, return metric dict."""
    model.bc_transformer.set_n_future_attend(n_attend)
    model.eval()

    n_buttons = model.action_out_tokens_to_logits.__self__ if False else None
    metrics = DirectActionMetrics(
        n_buttons=model.gamepad_direct_action_mapping.max_buttons,
        n_stick_axes=4,
        stick_bins=model.gamepad_direct_action_mapping.n_stick_bins,
    )

    total_action_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        batch = _move_batch(batch, device)
        with torch.no_grad():
            actions_in, masked_labels, _ = model._create_target_and_masked_labels(batch)

        frames = model._normalize_frames(batch.frames)
        action_embeddings_in = model.action_in_to_tokens(actions_in)

        action_out_embeddings, _, future_vision_pred, auxiliary_losses, _ = (
            model.transformer_forward_function(
                frames,
                action_embeddings_in,
                batch.text_embeddings,
                skip_action_decoder=True,
            )
        )

        # Direct gamepad mapping: a_in_0 → [B, T, F, logits]
        a_in_0 = action_out_embeddings[:, :, 0, :]
        action_logits_all = model.action_out_tokens_to_logits(a_in_0)
        F_dec = model.n_direct_future_frames
        B = frames.shape[0]
        T = frames.shape[1]

        batch_action_loss = 0.0
        for i in range(F_dec):
            labels_i = model._build_future_masked_labels(masked_labels, i)
            action_logits_i = GamepadDirectActionLogits(
                buttons=action_logits_all.buttons[:, :, i, :],
                left_stick_x=action_logits_all.left_stick_x[:, :, i, :],
                left_stick_y=action_logits_all.left_stick_y[:, :, i, :],
                right_stick_x=action_logits_all.right_stick_x[:, :, i, :],
                right_stick_y=action_logits_all.right_stick_y[:, :, i, :],
            )
            aux_i = auxiliary_losses if i == 0 else {}
            loss_i, _, _ = model._calculate_action_losses_from_logits_eager(
                action_logits_i, labels_i, aux_i, B, T
            )
            batch_action_loss += loss_i.item()

            # Only accumulate metrics for frame 0 (current frame prediction)
            if i == 0:
                metrics.update(action_logits_i, labels_i)

        total_action_loss += batch_action_loss / F_dec
        total_batches += 1

        if (batch_idx + 1) % 20 == 0:
            logging.info(
                "  attend_k=%d  batch %d/%d  avg_loss=%.4f",
                n_attend, batch_idx + 1, n_batches,
                total_action_loss / total_batches,
            )

    result = metrics.compute()
    result["action_loss"] = round(total_action_loss / max(total_batches, 1), 4)
    result["n_batches"] = total_batches
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="StaMo co-training ablation: vary n_future_attend at inference time"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--data_folder", required=True, help="Dataset root directory")
    parser.add_argument("--n_batches", type=int, default=200, help="Batches per condition")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--attend_ks",
        type=int,
        nargs="+",
        default=[0, 1, 5, 10],
        help="n_future_attend values to evaluate",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default=None, help="Optional path to save results JSON")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load config and override data path
    config = load_config(args.config, LightningPolicyConfig)
    config.stage3_finetune.training_dataset.local_prefix = args.data_folder
    for val_cfg in config.stage3_finetune.validation_datasets:
        val_cfg.local_prefix = args.data_folder

    # Reduce dataloader overhead for validation
    val_cfg = config.stage3_finetune.validation_datasets[0]
    set_validation_dataset_cfg_to_single_thread(val_cfg, batch_size=args.batch_size)

    # Load model
    model = _load_model(args.checkpoint, config)
    model = model.to(device)
    model.eval()

    # Build dataloader via Stage3DataModule
    datamodule = Stage3DataModule(config)
    datamodule.setup("validate")
    val_loaders = datamodule.val_dataloader()
    if isinstance(val_loaders, list):
        val_loader = val_loaders[0]
    else:
        val_loader = val_loaders

    # Disable torch.compile for evaluation
    torch.compiler.set_stance("force_eager")

    N = model.bc_transformer.n_future_state_tokens
    attend_ks = [k for k in args.attend_ks if 0 <= k <= N]
    logging.info(
        "Running ablation: N=%d, attend_ks=%s, n_batches=%d", N, attend_ks, args.n_batches
    )

    all_results = {}
    for k in attend_ks:
        logging.info("=== Evaluating attend_k=%d ===", k)
        result = evaluate_condition(model, val_loader, k, args.n_batches, device)
        all_results[k] = result
        logging.info("  attend_k=%d results: %s", k, json.dumps(result, indent=2))

    # Print summary table
    print("\n" + "=" * 80)
    print(f"StaMo Co-Training Ablation  (checkpoint: {os.path.basename(args.checkpoint)})")
    print("=" * 80)
    header = f"{'attend_k':>10} | {'action_loss':>12} | {'btn_acc':>9} | {'btn_f1':>9} | {'stick_mae':>10} | {'stick_dir_acc':>14}"
    print(header)
    print("-" * len(header))
    for k in attend_ks:
        r = all_results[k]
        print(
            f"{k:>10} | {r.get('action_loss', float('nan')):>12.4f} | "
            f"{r.get('button_accuracy', float('nan')):>9.4f} | "
            f"{r.get('button_f1_macro', float('nan')):>9.4f} | "
            f"{r.get('stick_mae', float('nan')):>10.4f} | "
            f"{r.get('stick_direction_accuracy', float('nan')):>14.4f}"
        )
    print("=" * 80)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
        logging.info("Saved results to %s", args.output_json)


if __name__ == "__main__":
    main()
