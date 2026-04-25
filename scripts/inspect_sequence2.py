"""
inspect_sequence2.py — 单独评估第二条 sequence 的 TF vs AR 输出对比

目的：
  排查 TF 和 AR 在多 sequence 时后 200 帧出现巨大差异的根因。
  本脚本跳过第一条 sequence，只对第二条 sequence 同时运行：
    · TF (teacher-forcing)：整条 sequence 一次性 forward，GT action 作为输入
    · AR (autoregressive)：逐帧 online_forward，模型自己的输出写入 KV cache
  并逐帧打印 pred/gt 对比，以及 TF 和 AR 之间的 logit 差异。

用法:
  python scripts/inspect_sequence2.py \
    --checkpoint_path output_20260420/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00100000.ckpt \
    --config_path config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml \
    --data_folder cuphead_one_level_3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ── 确保项目根目录在 sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig, ValidationDatasetConfig
from elefant.policy_model.stage3_finetune import (
    Stage3DataModule,
    Stage3FutureVisionLightning,
    Stage3LabelledBCLightning,
    GamepadDirectActionLogits,
)
from elefant.policy_model.validation_action_direct import (
    move_batch_to_device,
    set_validation_dataset_cfg_to_single_thread,
)
from elefant.policy_model.validation_autoregressive_action_direct import (
    _load_model,
    _ensure_validation_datasets,
)


# ── TF 单条 sequence forward ───────────────────────────────────────────────────

def tf_forward_sequence(model, batch_cuda):
    """
    对一条 sequence 做 teacher-forcing forward，返回逐帧 logits。
    返回: GamepadDirectActionLogits，每个字段 shape [1, T, ...]
    """
    actions_in, masked_labels, _ = model._create_target_and_masked_labels(batch_cuda)
    frames = model._normalize_frames(batch_cuda.frames)
    B, T = frames.shape[:2]
    action_embeddings_in = model.action_in_to_tokens(actions_in)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if isinstance(model, Stage3FutureVisionLightning):
            action_out_embeddings, _, _, _, _ = model.transformer_forward_function(
                frames,
                action_embeddings_in,
                batch_cuda.text_embeddings,
                skip_action_decoder=True,
            )
        else:
            action_out_embeddings, _, _, _ = model.transformer_forward_function(
                frames,
                action_embeddings_in,
                batch_cuda.text_embeddings,
                skip_action_decoder=True,
            )
        # action_out_embeddings: [B, T, 1, D]，取 a_in^0
        # action_out_tokens_to_logits 返回 [B, T, F, ...]，取 frame 0 用于当前帧评估
        action_logits_all = model.action_out_tokens_to_logits(action_out_embeddings[:, :, 0, :])
        from elefant.policy_model.stage3_finetune import GamepadDirectActionLogits
        action_logits = GamepadDirectActionLogits(
            buttons=action_logits_all.buttons[:, :, 0, :],
            left_stick_x=action_logits_all.left_stick_x[:, :, 0, :],
            left_stick_y=action_logits_all.left_stick_y[:, :, 0, :],
            right_stick_x=action_logits_all.right_stick_x[:, :, 0, :],
            right_stick_y=action_logits_all.right_stick_y[:, :, 0, :],
        )

    return action_logits, masked_labels  # [1, T, ...]


# ── AR 单条 sequence 逐帧推理 ──────────────────────────────────────────────────

def ar_forward_sequence(model, batch_cuda, reset_each_frame: bool = False):
    """
    对一条 sequence 做自回归逐帧推理。

    reset_each_frame=False（默认）：正常 AR，KV cache 跨帧累积。
    reset_each_frame=True：每帧独立推理，KV cache 和 idx 每帧重置为 0，
        相当于"无历史"单帧推理，用于排查 KV cache 是否是动作粘滞的根因。

    返回:
        pred_list  — list[T] of [n_buttons+4]，0/1 按钮 + bin index 摇杆
        logit_list — list[T] of [n_buttons] float32，button raw logits（sigmoid 前）
                     通过 forward hook 捕获 direct_button_head 的输出。
    """
    B = batch_cuda.frames.shape[0]
    T = batch_cuda.frames.shape[1]
    assert B == 1, "AR forward 仅支持 batch_size=1"

    frames = batch_cuda.frames
    text_embed = batch_cuda.text_embeddings

    # ── 注册 hook 捕获 direct_button_head 的原始输出（sigmoid 前）──────────────
    _captured_logits = []
    n_buttons_local = model.n_direct_buttons
    def _btn_hook(module, input, output):
        # output: [B, 1, F*n_buttons]；取 frame 0 的 n_buttons 个 logits
        _captured_logits.append(
            output.detach().float()[0, 0, :n_buttons_local].cpu()
        )

    hook_handle = model.direct_button_head.register_forward_hook(_btn_hook)

    kv_cache_state = model.bc_transformer.init_kv_cache_state()
    idx = torch.tensor(0, dtype=torch.int64, device=model.device)

    pred_list = []

    try:
        for t in range(T):
            if reset_each_frame:
                kv_cache_state = model.bc_transformer.init_kv_cache_state()
                idx = torch.tensor(0, dtype=torch.int64, device=model.device)

            frame_t = frames[0, t]
            text_t = text_embed[0, t] if text_embed is not None else None

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sampled_action, idx, kv_cache_state = model.online_kv_cache_predict(
                    frame_t,
                    idx=idx,
                    kv_cache_state=kv_cache_state,
                    text_tokens_embed=text_t,
                    compile=False,
                )

            pred_list.append(sampled_action.squeeze(0).cpu())
    finally:
        hook_handle.remove()

    return pred_list, _captured_logits  # list[T] of [n_buttons+4], list[T] of [n_buttons]


# ── 打印逐帧对比 ───────────────────────────────────────────────────────────────

def print_frame_comparison(
    tf_logits: GamepadDirectActionLogits,
    ar_preds: list,
    ar_logits: list,
    masked_labels: torch.Tensor,
    n_buttons: int,
    n_stick_bins: int,
    max_frames: int = 200,
    seq_index: int = 1,
    show_logits: bool = False,
):
    """
    逐帧打印 TF pred / AR pred / GT，以及可选的 button logits 对比。

    show_logits=True 时，对每个有差异的帧额外打印一行：
        TF_logits[bi] vs AR_logits[bi]（sigmoid 前的原始值），
        方便判断是"logit 没动"还是"logit 动了但没过 0.5 阈值"。

    masked_labels: [1, T, n_buttons+4]
    ar_logits: list[T] of [n_buttons] float32，由 forward hook 捕获
    """
    T = min(tf_logits.buttons.shape[1], len(ar_preds), max_frames)
    stick_center = n_stick_bins // 2

    # TF pred（从 logits 解码）
    tf_btn_pred = (tf_logits.buttons[0].sigmoid() > 0.5).long().cpu()  # [T, n_buttons]
    tf_lx_pred  = tf_logits.left_stick_x[0].argmax(-1).cpu()           # [T]
    tf_ly_pred  = tf_logits.left_stick_y[0].argmax(-1).cpu()
    tf_rx_pred  = tf_logits.right_stick_x[0].argmax(-1).cpu()
    tf_ry_pred  = tf_logits.right_stick_y[0].argmax(-1).cpu()

    # GT
    gt = masked_labels[0].cpu()  # [T, n_buttons+4]

    print(f"\n{'='*80}")
    print(f"逐帧对比 (T={T})")
    print(f"{'='*80}")
    print(f"{'帧':>4}  {'TF_btn':>20}  {'AR_btn':>20}  {'GT_btn':>20}  "
          f"{'TF_stk':>12}  {'AR_stk':>12}  {'GT_stk':>12}  "
          f"{'TF≠AR':>6}  {'TF≠GT':>6}  {'AR≠GT':>6}  {'STK_DIFF':>8}")
    print("-" * 130)

    btn_diff_tf_ar_frames = []
    btn_diff_tf_gt_frames = []
    stk_diff_frames = []

    for t in range(T):
        ar_pred = ar_preds[t]  # [n_buttons+4]
        ar_btn = ar_pred[:n_buttons]
        ar_stk = ar_pred[n_buttons:n_buttons+4]

        tf_btn = tf_btn_pred[t]
        tf_stk = torch.stack([tf_lx_pred[t], tf_ly_pred[t], tf_rx_pred[t], tf_ry_pred[t]])

        gt_btn = gt[t, :n_buttons]
        gt_stk = gt[t, n_buttons:n_buttons+4]

        tf_ar_btn_diff = (tf_btn != ar_btn).sum().item()
        tf_gt_btn_diff = (tf_btn != gt_btn).sum().item()
        ar_gt_btn_diff = (ar_btn != gt_btn).sum().item()
        tf_ar_stk_diff = (tf_stk.float() - ar_stk.float()).abs().mean().item()

        if tf_ar_btn_diff > 0:
            btn_diff_tf_ar_frames.append(t)
        if tf_gt_btn_diff > 0:
            btn_diff_tf_gt_frames.append(t)
        if tf_ar_stk_diff > 0.5:
            stk_diff_frames.append(t)

        # 有任何差异的帧，或每 20 帧打印一次
        has_diff = tf_ar_btn_diff > 0 or tf_gt_btn_diff > 0 or ar_gt_btn_diff > 0 or tf_ar_stk_diff > 0.5
        if has_diff or t % 20 == 0:
            tf_btn_str = "".join(str(x.item()) for x in tf_btn)
            ar_btn_str = "".join(str(x.item()) for x in ar_btn)
            gt_btn_str = "".join(str(x.item()) for x in gt_btn)
            tf_stk_str = f"{tf_stk.tolist()}"
            ar_stk_str = f"{ar_stk.tolist()}"
            gt_stk_str = f"{gt_stk.tolist()}"
            marker = " <<<" if has_diff else ""
            print(f"{t:>4}  {tf_btn_str:>20}  {ar_btn_str:>20}  {gt_btn_str:>20}  "
                  f"{tf_stk_str:>12}  {ar_stk_str:>12}  {gt_stk_str:>12}  "
                  f"{tf_ar_btn_diff:>6}  {tf_gt_btn_diff:>6}  {ar_gt_btn_diff:>6}  "
                  f"{tf_ar_stk_diff:>8.3f}{marker}")

            # 打印 button logits（仅在有差异帧 + show_logits 时）
            if show_logits and has_diff:
                tf_raw = tf_logits.buttons[0, t].float().cpu()  # [n_buttons]
                ar_raw = ar_logits[t]                            # [n_buttons] or None
                tf_sig = tf_raw.sigmoid()
                ar_sig = ar_raw.sigmoid() if ar_raw is not None else None

                # 只打印 TF 和 AR 预测不同的按钮，或 GT=1 的按钮
                gt_btn_mask = (gt_btn != -100)
                for bi in range(n_buttons):
                    if not gt_btn_mask[bi]:
                        continue
                    tf_b = tf_btn[bi].item()
                    ar_b = ar_btn[bi].item() if ar_raw is not None else "?"
                    gt_b = gt_btn[bi].item()
                    # 只打印有差异或 GT=1 的按钮
                    if tf_b != ar_b or gt_b == 1:
                        tf_l = f"{tf_raw[bi].item():+.3f}(σ={tf_sig[bi].item():.3f})"
                        ar_l = (f"{ar_raw[bi].item():+.3f}(σ={ar_sig[bi].item():.3f})"
                                if ar_raw is not None else "N/A")
                        flag = "DIFF" if tf_b != ar_b else "    "
                        print(f"       btn[{bi:2d}] TF={tf_l}  AR={ar_l}  GT={gt_b}  {flag}")

    print(f"\n{'='*80}")
    print(f"TF vs AR 差异帧统计:")
    print(f"  TF≠AR 按钮差异帧数: {len(btn_diff_tf_ar_frames)} / {T}")
    print(f"  TF≠GT 按钮差异帧数: {len(btn_diff_tf_gt_frames)} / {T}")
    print(f"  摇杆 TF-AR MAE>0.5 的帧数: {len(stk_diff_frames)} / {T}")
    if btn_diff_tf_ar_frames:
        print(f"  TF≠AR 差异帧索引 (前20): {btn_diff_tf_ar_frames[:20]}")
    if btn_diff_tf_gt_frames:
        print(f"  TF≠GT 差异帧索引 (前20): {btn_diff_tf_gt_frames[:20]}")
    if stk_diff_frames:
        print(f"  摇杆差异帧索引 (前20): {stk_diff_frames[:20]}")

    # 汇总指标
    tf_btn_acc = (tf_btn_pred[:T] == gt[:T, :n_buttons]).float().mean().item()
    ar_btn_acc_list = [(ar_preds[t][:n_buttons] == gt[t, :n_buttons]).float().mean().item() for t in range(T)]
    ar_btn_acc = sum(ar_btn_acc_list) / len(ar_btn_acc_list)

    tf_stk_mae = (
        torch.stack([tf_lx_pred[:T].float(), tf_ly_pred[:T].float(),
                     tf_rx_pred[:T].float(), tf_ry_pred[:T].float()], dim=1)
        - gt[:T, n_buttons:n_buttons+4].float()
    ).abs().mean().item()
    ar_stk_mae = sum(
        (ar_preds[t][n_buttons:n_buttons+4].float() - gt[t, n_buttons:n_buttons+4].float()).abs().mean().item()
        for t in range(T)
    ) / T

    tf_btn_exact = sum(
        1 for t in range(T) if (tf_btn_pred[t] == gt[t, :n_buttons]).all().item()
    ) / T
    ar_btn_exact = sum(
        1 for t in range(T) if (ar_preds[t][:n_buttons] == gt[t, :n_buttons]).all().item()
    ) / T

    print(f"\n汇总指标 (sequence {seq_index+1}, T={T}):")
    print(f"  {'':12}  {'button_acc':>12}  {'btn_exact':>10}  {'stick_mae':>10}")
    print(f"  {'TF vs GT':12}  {tf_btn_acc:>12.4f}  {tf_btn_exact:>10.4f}  {tf_stk_mae:>10.4f}")
    print(f"  {'AR vs GT':12}  {ar_btn_acc:>12.4f}  {ar_btn_exact:>10.4f}  {ar_stk_mae:>10.4f}")
    print(f"{'='*80}")


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
        # default="output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_v350335326/stage3_finetune/checkpoint-step=00080000.ckpt")
        #default="output_20260420/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00100000.ckpt")
        default="output_20260425/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_onelevel3/stage3_finetune/checkpoint-step=00030000.ckpt")
    parser.add_argument("--config_path", type=str,
        default="config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml")
    parser.add_argument("--data_folder", type=str, default="cuphead_one_level_3")
    parser.add_argument("--seq_index", type=int, default=1,
        help="要检查的 sequence 索引（0-based），默认 1 即第二条 sequence")
    parser.add_argument("--max_frames", type=int, default=200,
        help="最多打印多少帧的对比（默认 200）")
    parser.add_argument("--reset_each_frame", action="store_true",
        help="每帧重置 KV cache 和 idx，屏蔽历史上下文，验证动作粘滞是否由 KV cache 引起")
    parser.add_argument("--show_logits", action="store_true",
        help="对有差异的帧打印 TF/AR button raw logits（sigmoid 前），观察 logit 是否真的没动")
    args = parser.parse_args()
    print(f"args: {args}")

    t0 = time.time()
    config = load_config(args.config_path, LightningPolicyConfig)

    config.stage3_finetune.training_dataset.local_prefix = args.data_folder
    _ensure_validation_datasets(config, args.data_folder)

    BATCH_SIZE = 1
    for i in range(len(config.stage3_finetune.validation_datasets)):
        config.stage3_finetune.validation_datasets[i] = set_validation_dataset_cfg_to_single_thread(
            config.stage3_finetune.validation_datasets[i], batch_size=BATCH_SIZE
        )

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = _load_model(args.checkpoint_path, config)
    model.eval()
    model = model.to("cuda")

    assert model._use_gamepad_direct_mapping(), "需要 gamepad_direct 映射"

    if hasattr(model, "bc_transformer") and hasattr(model.bc_transformer, "block_mask_to_device"):
        model.bc_transformer.block_mask_to_device(model.device)

    # AR 需要的 RoPE + KV cache 初始化
    max_virtual_idx = config.shared.n_seq_timesteps * model.bc_transformer.step_size
    model.bc_transformer.rebuild_rope_cache(max_virtual_idx)
    model.bc_transformer.setup_kv_cache(batch_size=1, device=model.device)

    if hasattr(model, "action_mapping"):
        _orig_make_empty = model.action_mapping.make_empty_action
        _device = model.device
        def _make_empty_action_cuda(T: int):
            return _orig_make_empty(T).to(_device)
        model.action_mapping.make_empty_action = _make_empty_action_cuda

    n_buttons = model.n_direct_buttons
    n_stick_bins = model.n_direct_stick_bins
    print(f"模型加载完成 ({time.time()-t0:.1f}s)  n_buttons={n_buttons}  n_stick_bins={n_stick_bins}")

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    datamodule = Stage3DataModule(config)
    datamodule.setup("stage3_finetune")
    val_dataloaders = datamodule.val_dataloader()
    del datamodule

    val_set_name = list(val_dataloaders.keys())[0]
    val_dataloader = val_dataloaders[val_set_name]
    print(f"使用验证集: {val_set_name}")

    # ── 跳到目标 sequence ─────────────────────────────────────────────────────
    target_batch = None
    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx == args.seq_index:
            target_batch = batch
            print(f"\n已获取 sequence index={args.seq_index} (batch_idx={batch_idx})")
            break
        else:
            print(f"  跳过 sequence {batch_idx}")

    if target_batch is None:
        print(f"ERROR: 数据集中不足 {args.seq_index+1} 条 sequence")
        return

    batch_cuda = move_batch_to_device(target_batch, "cuda")
    T = batch_cuda.frames.shape[1]
    print(f"Sequence shape: frames={batch_cuda.frames.shape}  T={T}")

    # ── TF forward ────────────────────────────────────────────────────────────
    print("\n[1/2] 运行 TF (teacher-forcing) forward...")
    tf_logits, masked_labels = tf_forward_sequence(model, batch_cuda)
    print(f"  TF logits.buttons shape: {tf_logits.buttons.shape}")

    # ── AR forward ────────────────────────────────────────────────────────────
    mode_str = "reset_each_frame（无历史）" if args.reset_each_frame else "正常 AR（KV cache 累积）"
    print(f"\n[2/2] 运行 AR ({mode_str}) 逐帧推理...")
    ar_preds, ar_logits = ar_forward_sequence(
        model, batch_cuda, reset_each_frame=args.reset_each_frame
    )
    print(f"  AR preds: {len(ar_preds)} 帧")

    # ── 逐帧对比打印 ──────────────────────────────────────────────────────────
    print_frame_comparison(
        tf_logits=tf_logits,
        ar_preds=ar_preds,
        ar_logits=ar_logits,
        masked_labels=masked_labels,
        n_buttons=n_buttons,
        n_stick_bins=n_stick_bins,
        max_frames=args.max_frames,
        seq_index=args.seq_index,
        show_logits=args.show_logits,
    )

    print(f"\n总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
