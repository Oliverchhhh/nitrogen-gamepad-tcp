"""
KV cache test for PolicyFutureCausalTransformer.online_forward().
Verifies:
  1. No TypeError from direct_action_head_fn kwarg (the stash fix)
  2. KV cache grows correctly each step (shape[2] += step_size)
  3. idx advances correctly (idx += step_size each step)
  4. sampled_action has the right shape
"""
import sys, torch, logging
logging.basicConfig(level=logging.WARNING)

CKPT = "checkpoints/202604261721_150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_v350335326/checkpoint-step=00200000.ckpt"
CONFIG = "config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml"
N_STEPS = 210  # 200 to fill cache + 10 to verify rolling eviction

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.stage3_finetune import Stage3FutureVisionLightning

print("Loading config...")
config = load_config(CONFIG, LightningPolicyConfig)

print("Loading model...")
model = Stage3FutureVisionLightning(config=config, inference_mode=True)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
if incompatible.missing_keys:
    print(f"  missing keys: {incompatible.missing_keys[:3]} ...")
if incompatible.unexpected_keys:
    print(f"  unexpected keys: {incompatible.unexpected_keys[:3]} ...")

model = model.to("cuda", dtype=torch.bfloat16)
model.eval()
print("Model loaded OK\n")

# --- step_size: what idx should advance by each frame ---
bc = model.bc_transformer
step_size = (
    bc.image_tokenizer.get_n_img_tokens()
    + bc.text_token_size
    + bc.config.n_thinking_tokens
    + 2                              # state_out + action_out (Future variant)
    + bc.config.n_action_tokens
)
print(f"Expected step_size per frame = {step_size}")
print(f"  n_img={bc.image_tokenizer.get_n_img_tokens()}, n_text={bc.text_token_size}, "
      f"n_think={bc.config.n_thinking_tokens}, n_action={bc.config.n_action_tokens}")
print(f"  zero_action_input={bc.config.zero_action_input}, n_transformer_layers={bc.config.n_transformer_layers}")

# --- init KV cache ---
# RoPE cache must cover all positions we'll visit (same as inference.py's rebuild_rope_cache call)
bc.rebuild_rope_cache(N_STEPS * step_size + step_size)
bc.block_mask_to_device(torch.device("cuda"))
bc.setup_kv_cache(batch_size=1, device=torch.device("cuda"))
max_seq_len = bc.kv_cache.max_seq_len
print(f"  max_seq_len={max_seq_len} (cache fills after {max_seq_len // step_size} steps)")
print()
kv_cache_state = bc.init_kv_cache_state()
idx = torch.tensor(0, dtype=torch.int64, device="cuda")
H, W = config.shared.frame_height, config.shared.frame_width

print(f"{'Step':>4}  {'idx_before':>10}  {'idx_after':>9}  {'idx_delta':>9}  "
      f"{'kv_len_before':>13}  {'kv_len_after':>12}  {'kv_delta':>8}  {'action_shape'}")
print("-" * 100)

dummy_text = torch.zeros(
    1, bc.text_token_size, bc.text_tokenizer_embed_dim,
    device="cuda", dtype=torch.bfloat16,
)

errors = 0
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for t in range(N_STEPS):
        dummy_frame = torch.zeros(3, H, W, device="cuda", dtype=torch.uint8)
        kv_len_before = kv_cache_state[0].k_cache.shape[2]
        idx_before = int(idx.item())
        is_rolling = kv_len_before >= max_seq_len

        sampled_action, idx, kv_cache_state = model.online_kv_cache_predict(
            dummy_frame,
            idx=idx,
            kv_cache_state=kv_cache_state,
            text_tokens_embed=dummy_text,
            compile=False,
        )

        idx_after = int(idx.item())
        kv_len_after = kv_cache_state[0].k_cache.shape[2]
        kv_delta = kv_len_after - kv_len_before
        idx_delta = idx_after - idx_before
        action_shape = tuple(sampled_action.shape)

        # Growing phase: cache should grow by step_size each step
        # Rolling phase: cache length stays at max_seq_len (evict step_size, append step_size)
        expected_kv_delta = 0 if is_rolling else step_size
        expected_kv_len_after = max_seq_len if is_rolling else kv_len_before + step_size

        ok_idx = "✓" if idx_delta == step_size else f"✗ expected {step_size}"
        ok_kv  = "✓" if kv_delta == expected_kv_delta else f"✗ expected {expected_kv_delta}"
        ok_len = "✓" if kv_len_after == expected_kv_len_after else f"✗ expected {expected_kv_len_after}"

        if idx_delta != step_size or kv_delta != expected_kv_delta or kv_len_after != expected_kv_len_after:
            errors += 1

        # Only print first/last few growing steps and all rolling steps
        if t < 3 or t >= max_seq_len // step_size - 2:
            mode = "[ROLL]" if is_rolling else "[GROW]"
            print(f"{t:>4} {mode}  {idx_before:>10}→{idx_after:<9}  Δidx={idx_delta}{ok_idx}  "
                  f"kv {kv_len_before:>6}→{kv_len_after:<6} Δkv={kv_delta:>3}{ok_kv} {ok_len}  act={action_shape}")
        elif t == 3:
            print(f"  ... (steps 3–{max_seq_len // step_size - 3} omitted, growing phase) ...")

print()
if errors == 0:
    print("✓ All checks passed. KV cache grow and rolling eviction both correct.")
else:
    print(f"✗ {errors} step(s) failed.")
