"""
Autoregressive Validation — direct action paradigm (non-autoregressive decoder).

【验证范式对比】
  - teacher-forcing (validation_action_direct.py):
      每步将 GT action 作为输入写入 KV cache，评估"给定完美历史时的预测能力"。
      优点：消除误差累积，衡量模型上限；缺点：与真实推理路径不一致。
  - autoregressive (本文件):
      每步将模型自己采样的 action 写入 KV cache，评估"实际推理时的表现"。
      与 inference.py 路径完全一致，能反映部署时的真实误差累积效果。

【direct action 范式】
  - 不使用 ActionDecoder 的自回归 token 解码，而是：
      a_in^0 (第一个 future action token 的输出) → MLP → 2 个独立 head：
        · direct_button_head:  sigmoid > 0.5 → 0/1 二值（BCE 训练）
        · direct_stick_head:   argmax → bin index（CE 训练，4 轴各 n_stick_bins 类）
  - 无扳机 head（Cuphead 游戏不使用扳机）。
  - zero_action_input=True：Pass 2 被跳过，KV cache 中 action 槽始终为零向量，
    消除 teacher-forcing 与 AR 推理之间的分布差异。

【整体流程】
  1. 加载 checkpoint → 模型 eval 模式 → 初始化 KV cache
  2. 遍历验证集，每条序列独立重置 KV cache（idx=0）
  3. 逐帧调用 online_kv_cache_predict：
       frame[t] + text_embed[t] → Transformer (Pass 1 only) → a_in^0
       → _direct_head_fn → sampled_action [1, n_buttons+4]
  4. sampled_action vs GT action → AutoregressiveDirectMetrics.update_step()
  5. 所有序列结束后 compute() 汇总指标，打印并保存 JSON

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
#
# 设计原则：
#   · 所有指标均基于"index tensor"（非 logits），与 online_kv_cache_predict 输出对齐。
#   · GT 中 -100 表示该帧/该列无标注，跳过不计入统计。
#   · _RunningMean 用于在线累积均值，避免存储所有帧数据。
# ---------------------------------------------------------------------------

@dataclass
class _RunningMean:
    """在线累积均值，支持批量 update（sum_value 为一批的总和，n 为样本数）。"""
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

    【按钮指标】(binary BCE 范式，pred/gt 均为 0/1)
        button_accuracy      per-button 准确率（pred > 0 vs gt > 0）
        button_exact_match   一帧内所有按钮全部正确的比例
        button_f1_macro      宏平均 F1（各按钮 F1 的算术均值）

    【摇杆指标】(4 轴 CE 范式，pred/gt 均为 bin index)
        stick_mae            bin MAE（4 轴平均）
        stick_rmse           bin RMSE（4 轴平均）
        stick_direction_acc  三分类方向准确率（负/零/正，4 轴平均）
        stick_deadzone_acc   GT 在死区时预测也在死区的比例（4 轴平均）

    注：direct 范式无扳机 head，不计算 trigger 指标。
    """

    STICK_AXES = ["left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"]

    def __init__(self, n_buttons: int, n_stick_bins: int):
        self.n_buttons = n_buttons
        self.n_stick_bins = n_stick_bins
        # 中心 bin（对应摇杆归零位置）
        self.stick_center = n_stick_bins // 2
        # 死区半径：bins <= 3 时死区仅为中心点，否则为 ±1
        self.stick_dz_half = 0 if n_stick_bins <= 3 else 1

        # ---- 按钮累积统计 ----
        self.btn_correct = _RunningMean()       # 所有按钮的正确预测总数
        self.btn_exact_match = _RunningMean()   # 全帧完全匹配次数
        # 每个按钮的 TP/FP/FN，用于计算 per-button F1
        self._btn_tp = [0] * n_buttons
        self._btn_fp = [0] * n_buttons
        self._btn_fn = [0] * n_buttons

        # ---- 摇杆累积统计（4 轴各独立） ----
        self.stick_mae = [_RunningMean() for _ in range(4)]       # 绝对误差
        self.stick_sq_err = [_RunningMean() for _ in range(4)]    # 平方误差（用于 RMSE）
        self.stick_dir_correct = [_RunningMean() for _ in range(4)]  # 方向分类正确
        self.stick_dz_correct = [_RunningMean() for _ in range(4)]   # 死区命中率

    @staticmethod
    def _to_direction(val: int, center: int, dz: int) -> int:
        """将 bin index 映射为三分类方向：-1（负）/ 0（死区）/ 1（正）。"""
        if val < center - dz:
            return -1
        elif val > center + dz:
            return 1
        return 0

    def update_step(self, pred_action: torch.Tensor, gt_action: torch.Tensor):
        """
        更新单帧指标。

        pred_action: [n_buttons + 4]  — 模型采样的 action index（direct head 输出）
                     前 n_buttons 列为按钮 0/1，后 4 列为摇杆 bin index
        gt_action:   [n_buttons + 4]  — GT label（-100 表示该列无标注，跳过）
        """
        k = self.n_buttons

        # ---- 按钮（binary 0/1） ----
        pred_btn = pred_action[:k]
        gt_btn = gt_action[:k]
        valid = gt_btn != -100  # 过滤无标注列

        if valid.any():
            # direct head 输出已经是 0/1，gt 也是 0/1，直接比较
            correct = pred_btn[valid] == gt_btn[valid]
            self.btn_correct.update(correct.float().sum().item(), correct.numel())

            # exact match：仅当所有按钮均有标注时才计算（避免部分标注干扰）
            if valid.all():
                em = (pred_btn == gt_btn).all().float().item()
                self.btn_exact_match.update(em, 1)

            # 逐按钮累积 TP/FP/FN，用于后续 F1 计算
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

        # ---- 摇杆（4 轴，bin index） ----
        for i in range(4):
            col = k + i
            if col >= gt_action.shape[0] or gt_action[col] == -100:
                continue
            p_val = pred_action[col].float()
            g_val = gt_action[col].float()
            ae = (p_val - g_val).abs().item()
            self.stick_mae[i].update(ae, 1)
            self.stick_sq_err[i].update(ae ** 2, 1)

            # 方向准确率：将 bin index 映射为 -1/0/1 后比较
            d_pred = self._to_direction(pred_action[col].item(), self.stick_center, self.stick_dz_half)
            d_gt = self._to_direction(gt_action[col].item(), self.stick_center, self.stick_dz_half)
            self.stick_dir_correct[i].update(float(d_pred == d_gt), 1)

            # 死区准确率：仅在 GT 落在死区时统计，衡量模型"停手"能力
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
        """汇总所有帧的统计，返回指标字典。"""
        result: Dict[str, Any] = {}

        # ---- 按钮汇总 ----
        result["button_accuracy"] = self.btn_correct.mean
        result["button_exact_match"] = self.btn_exact_match.mean
        result["button_sample_count"] = self.btn_correct.count

        # 宏平均 F1：各按钮 F1 的算术均值（对稀疏按钮更公平）
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

        # ---- 摇杆汇总 ----
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

        # 4 轴平均汇总指标
        result["stick_mae"] = round(sum(mae_vals) / len(mae_vals), 4) if mae_vals else None
        result["stick_rmse"] = round(sum(rmse_vals) / len(rmse_vals), 4) if rmse_vals else None
        result["stick_direction_accuracy"] = round(sum(dir_vals) / len(dir_vals), 4) if dir_vals else None
        result["stick_deadzone_accuracy"] = round(sum(dz_vals) / len(dz_vals), 4) if dz_vals else None
        result["per_stick"] = per_stick

        return result


# ---------------------------------------------------------------------------
# 模型加载
#
# 根据 config 中是否有 state_target_tokenizer 或 use_precomputed_vision_features
# 来区分 Stage3FutureVisionLightning（含未来帧预测头）和 Stage3LabelledBCLightning（纯 BC）。
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, config: LightningPolicyConfig):
    """
    加载 Lightning 模型 checkpoint。

    · 若 config 包含未来帧相关字段（state_target_tokenizer 或 use_precomputed_vision_features），
      使用 Stage3FutureVisionLightning，并以 strict=False 加载（允许部分权重不匹配）。
    · 否则使用 Stage3LabelledBCLightning.load_from_checkpoint（严格加载）。
    · 两者均以 inference_mode=True 初始化，禁用训练专用组件（如 EMA、loss head 等）。
    """
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
    确保 config.stage3_finetune.validation_datasets 非空。

    部分本地 config 只定义了 training_dataset（或旧版 validation_dataset），
    而 AR 验证需要 validation_datasets 列表。
    此函数在列表为空时，将 training_dataset 的字段复制一份作为 fallback，
    并为每个 dataset 补全 validation_name 和 local_prefix（若命令行传入了 data_folder）。
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


def _enable_full_causal_mask_in_ar(model) -> int:
    """
    Disable per-layer block masks and force dense causal attention.

    Returns:
        Number of attention layers switched from block-mask mode to full-causal mode.
    """
    switched_layers = 0
    if not hasattr(model, "bc_transformer"):
        return switched_layers
    if not hasattr(model.bc_transformer, "_transformer"):
        return switched_layers

    layers = getattr(model.bc_transformer._transformer, "transformer_layers", [])
    for layer in layers:
        if not hasattr(layer, "self_attention"):
            continue
        sa = layer.self_attention
        if getattr(sa, "block_mask", None) is not None:
            switched_layers += 1
        # Force dense causal path in SelfAttention.forward(...):
        # if flex_attention_mask is None -> scaled_dot_product_attention with tril mask
        sa.block_mask = None
        sa._cached_decode_mask = None
        sa.is_causal = True

    return switched_layers


# ---------------------------------------------------------------------------
# 主验证逻辑
#
# 核心设计：完全复现 inference.py 的推理路径，唯一区别是将采样结果与 GT 对比计算指标。
# ---------------------------------------------------------------------------

def run_autoregressive_direct_validation(
    checkpoint_path: str,
    config_path: str,
    data_folder: Optional[str] = None,
    n_sequences: int = 50,
    use_full_causal_mask: bool = False,
):
    """
    自回归验证主函数 — direct action 范式。

    【每条序列的处理流程】
      1. 重置 KV cache（idx=0），模拟从第 0 帧开始的全新推理会话。
      2. 逐帧 t=0..T-1：
           frame[t] + text_embed[t]
             → online_kv_cache_predict（内部调用 bc_transformer.online_forward）
             → Pass 1: Transformer 前向，取 a_in^0 输出
             → _direct_head_fn: MLP → sigmoid/argmax → sampled_action [1, n_buttons+4]
             → (zero_action_input=True，跳过 Pass 2，KV cache 中 action 槽保持零向量)
             → idx 自增 step_size，KV cache 追加当前帧的 K/V
      3. sampled_action vs gt_actions[b, t, :n_buttons+4] → update_step()
      4. 所有序列结束后 compute() 汇总，打印并保存 JSON。

    【注意事项】
      · BATCH_SIZE=1：KV cache 当前仅支持 batch=1 的在线推理。
      · compile=True：使用 torch.compile 加速 _predict 内核（首次调用有编译开销）。
      · gt_actions 列顺序为 [buttons..., lx, ly, rx, ry, lt, rt]，
        direct pred 只有前 n_buttons+4 列（无扳机），取 gt[:n_buttons+4] 对齐。
    """
    t0 = time.time()
    config = load_config(config_path, LightningPolicyConfig)

    if data_folder:
        config.stage3_finetune.training_dataset.local_prefix = data_folder
    _ensure_validation_datasets(config, data_folder)

    # 强制 batch_size=1，单线程加载，避免 KV cache 多 batch 冲突
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

    # 确认使用 gamepad_direct 映射（本脚本不支持其他映射类型）
    assert model._use_gamepad_direct_mapping(), (
        "This script requires action_mapping_type='gamepad_direct'. "
        "Use validation_autoregressive.py for other mapping types."
    )

    # 默认使用 block mask；可选切换为完整 causal mask 做 AR 对照实验
    if use_full_causal_mask:
        switched = _enable_full_causal_mask_in_ar(model)
        print(
            f"[AR] full causal mask enabled: switched {switched} attention layers"
        )
    else:
        # 将 block mask 移到 GPU（部分模型使用 block-sparse attention）
        if hasattr(model, "bc_transformer") and hasattr(
            model.bc_transformer, "block_mask_to_device"
        ):
            model.bc_transformer.block_mask_to_device(model.device)

    # 初始化 RoPE 位置编码缓存和 KV cache 结构（与 inference.py 完全一致）
    # max_virtual_idx = 序列最大帧数 × 每帧 token 数（step_size）
    max_virtual_idx = config.shared.n_seq_timesteps * model.bc_transformer.step_size # 200 * 22 = 4400
    model.bc_transformer.rebuild_rope_cache(max_virtual_idx)
    model.bc_transformer.setup_kv_cache(batch_size=1, device=model.device) #每一层的precompute_decode_masks都指向index=34

    # make_empty_action 需要在 cuda 上创建 tensor，包装一层确保设备一致
    if hasattr(model, "action_mapping"):
        _orig_make_empty = model.action_mapping.make_empty_action
        _device = model.device
        def _make_empty_action_cuda(T: int):
            return _orig_make_empty(T).to(_device)
        model.action_mapping.make_empty_action = _make_empty_action_cuda

    n_buttons = model.n_direct_buttons       # 按钮数量（Cuphead 约 12 个）
    n_stick_bins = model.n_direct_stick_bins  # 摇杆离散 bin 数（config 中为 3）

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
        "mask_mode": "full_causal" if use_full_causal_mask else "block_mask",
    }
    total_seq_count = 0

    for val_set_name, val_dataloader in val_dataloaders.items():
        print(f"\n{'='*60}")
        print(f"AR-Direct Validation: {val_set_name}")
        print(f"{'='*60}")

        ar_metrics = AutoregressiveDirectMetrics(
            n_buttons=n_buttons,
            n_stick_bins=n_stick_bins,
        ) #指标实现

        seq_count = 0 #序列计数
        for batch_idx, batch in enumerate(val_dataloader):
            if seq_count >= n_sequences: #如果序列计数大于等于n_sequences，则跳出循环， 默认我们跑一遍也就是val目录内的视频数*6=seq数量
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

                # 每条序列独立重置 KV cache，模拟从第 0 帧开始的全新推理
                kv_cache_state = model.bc_transformer.init_kv_cache_state() #每层的KVCacheState对象包含k_cache和v_cache，(batch_size, num_kv_heads, 0, embed_size_per_head), [1, 16, 0, 64], 2维度动态可变
                idx = torch.tensor(0, dtype=torch.int64, device="cuda")

                for t in range(T):
                    frame_t = frames[b, t]  # [C, H, W]，单帧, [3,192,192]
                    # text_embed: [B, T, text_token_size, embed_dim]
                    # online_forward 期望 [text_token_size, embed_dim]，取 [b, t] 即可
                    text_t = (
                        text_embed[b, t]  # [text_token_size, embed_dim]
                        if text_embed is not None else None
                    ) #[1, 200, 1, 768], 全0

                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # 核心推理调用：与 inference.py 路径完全一致
                        # 内部流程：normalize → unsqueeze → online_forward (Pass 1 only)
                        #           → _direct_head_fn → sampled_action [1, n_buttons+4]
                        #           → idx += step_size，KV cache 追加当前帧 K/V
                        sampled_action, idx, kv_cache_state = model.online_kv_cache_predict(
                            frame_t,
                            idx=idx,
                            kv_cache_state=kv_cache_state,
                            text_tokens_embed=text_t,
                            compile=True,
                        )
                        # if kv_cache_state and len(kv_cache_state) > 0:
                        #     k_shape = tuple(kv_cache_state[0].k_cache.shape)
                        #     v_shape = tuple(kv_cache_state[0].v_cache.shape)
                            # print(
                            #     f"kv_cache layer0 k={k_shape} v={v_shape} idx={int(idx.item())}"
                            # )

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

    # 将结果保存到 checkpoint 同目录，文件名含 step 编号便于追踪
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
    parser.add_argument("--checkpoint_path", type=str, default="output_20260420/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00100000.ckpt")
    parser.add_argument("--config_path", type=str, default="config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml")
    parser.add_argument("--data_folder", type=str, default="cuphead_one_level_3",
                        help="覆盖 config 中的 local_prefix（数据集根目录）")
    parser.add_argument("--n_sequences", type=int, default=24,
                        help="验证的视频序列数量")
    parser.add_argument(
        "--full_causal_mask",
        action="store_true",
        help="禁用 block mask，改用完整 causal mask（用于 AR 对照实验）",
    )
    args = parser.parse_args()
    print(f"args: {args}")
    run_autoregressive_direct_validation(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        data_folder=args.data_folder,
        n_sequences=args.n_sequences,
        use_full_causal_mask=args.full_causal_mask,
    )


if __name__ == "__main__":
    main()
