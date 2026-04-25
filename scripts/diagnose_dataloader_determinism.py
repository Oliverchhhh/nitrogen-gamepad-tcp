"""
验证 shuffle=False 后，两次独立运行 val_dataloader 的序列顺序是否完全一致。
跑两次迭代，对比每个 batch 的帧内容（用 mean 作为指纹）。
"""
import torch
import logging
from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.stage3_finetune import Stage3DataModule
from elefant.policy_model.validation_action_direct import (
    move_batch_to_device, set_validation_dataset_cfg_to_single_thread,
)
from elefant.policy_model.validation_autoregressive_action_direct import _ensure_validation_datasets

CONFIG = "config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml"
DATA = "cuphead_one_level_3"
N_BATCHES = 4  # 只检查前 4 个 batch

logging.basicConfig(level=logging.WARNING)
config = load_config(CONFIG, LightningPolicyConfig)
config.stage3_finetune.training_dataset.local_prefix = DATA
_ensure_validation_datasets(config, DATA)

for ds in config.stage3_finetune.validation_datasets:
    set_validation_dataset_cfg_to_single_thread(ds, batch_size=1)

def collect_fingerprints(n_batches):
    """创建新的 datamodule，迭代前 n_batches 个 batch，返回帧均值指纹列表。"""
    dm = Stage3DataModule(config)
    dm.setup("stage3_finetune")
    val_dls = dm.val_dataloader()
    dl = list(val_dls.values())[0]

    fingerprints = []
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        batch_cuda = move_batch_to_device(batch, "cuda")
        # 用帧均值作为序列指纹
        fp = batch_cuda.frames.float().mean().item()
        fp_std = batch_cuda.frames.float().std().item()
        fingerprints.append((fp, fp_std))
        print(f"  batch {i}: frame_mean={fp:.6f}  frame_std={fp_std:.6f}")

    del dm
    return fingerprints

print(f"{'='*60}")
print(f"Run 1:")
print(f"{'='*60}")
fps1 = collect_fingerprints(N_BATCHES)

print(f"\n{'='*60}")
print(f"Run 2:")
print(f"{'='*60}")
fps2 = collect_fingerprints(N_BATCHES)

print(f"\n{'='*60}")
print(f"Comparison:")
print(f"{'='*60}")
all_match = True
for i, (f1, f2) in enumerate(zip(fps1, fps2)):
    match = abs(f1[0] - f2[0]) < 1e-5 and abs(f1[1] - f2[1]) < 1e-5
    status = "✅ MATCH" if match else "❌ DIFFER"
    print(f"  batch {i}: {status}  run1=({f1[0]:.6f},{f1[1]:.6f})  run2=({f2[0]:.6f},{f2[1]:.6f})")
    if not match:
        all_match = False

print(f"\n{'='*60}")
print(f"Result: {'DETERMINISTIC ✅' if all_match else 'NON-DETERMINISTIC ❌'}")
print(f"{'='*60}")
