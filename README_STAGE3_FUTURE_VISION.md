# 如何训练 Future 版 Stage3（Stage3FutureVisionLightning）

本文说明如何使用 **Stage3FutureVisionLightning** 训练带「未来帧视觉预测」的 Stage3 模型（Policy + Future Vision Head）。

---

## 1. 模型与任务

- **Stage3FutureVisionLightning**：在基础 Stage3 行为克隆（BC）之上，增加「从当前帧的 a⁰ 预测下一帧的全局视觉表征」的辅助任务。
- **PolicyFutureCausalTransformer**：在 `PolicyCausalTransformer` 上增加 `future_vision_head`，对每帧的 a⁰ 输出再做一个线性层，得到 `future_vision_pred`（形状 `[B, T, embed_dim]`）。
- **监督目标**：
  - 对 `t < T-1`：用**下一帧**经 `image_tokenizer` 后对空间维度 mean 得到的全局特征作为 target；
  - 对最后一帧 `t = T-1`：用**固定零向量**作为 target（dummy last frame），避免泄露序列结束信息。
- **损失**：`total_loss = action_loss + future_vision_loss_weight * future_vision_loss`，其中 `future_vision_loss` 为 MSE，默认权重 `future_vision_loss_weight = 0.1`（在代码中写死，未从配置读取）。

---

## 2. 环境与数据

- 与基础 Stage3 相同：Python 环境、CUDA、数据集格式（含 `.proto` 标注）。
- 数据集目录：默认使用项目根目录下的 `dataset`（可由 `--data_folder` 覆盖）。
- 视觉编码器：由配置文件中的 `shared.tokenizer` 决定；示例配置 `150M_local_dataset_future.yaml` 使用 **conv** tokenizer。若改用 dinov2/vjepa2，需在配置里改 `type` 及对应 tokenizer 配置。

---

## 3. 配置文件

- **推荐配置**：`config/policy_model/150M_local_dataset_future.yaml`
- **要点**：
  - `shared.output_path`: `output/policy_model/150M_future`（与基础 Stage3 输出分离）
  - `shared.n_seq_timesteps`: 200（与基础版一致）
  - `stage3_finetune.training_dataset.batch_size`: 2（可按显存改为 1）
  - `wandb.exp_name`: `150M-local-dataset-future`（便于与基础版区分）
- 其他优化器、验证间隔、保存步数等与基础 Stage3 相同，按需在 YAML 中修改。

---

## 4. 训练方式

### 4.1 使用脚本（推荐）

```bash
# 在项目根目录执行
./scripts/train_local_dataset_future.sh
```

脚本会：

- 检查 `dataset` 目录及 `.proto` 文件；
- 检查 `config/policy_model/150M_local_dataset_future.yaml`；
- 设置 `TORCHINDUCTOR_FX_GRAPH_CACHE`、`TORCHINDUCTOR_CACHE_DIR`（编译缓存）；
- 调用：`uv run elefant/policy_model/train_future.py --config config/policy_model/150M_local_dataset_future.yaml --data_folder dataset`

如需将日志写入文件，例如：

```bash
./scripts/train_local_dataset_future.sh 2>&1 | tee logs/train_future_$(date +%Y%m%d_%H%M%S).log
```

### 4.2 直接使用 Python 入口

```bash
uv run elefant/policy_model/train_future.py \
  --config config/policy_model/150M_local_dataset_future.yaml \
  --data_folder dataset
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（必填） |
| `--data_folder` | 数据集根目录，覆盖配置中的 `local_prefix` |
| `--no_compile` | 关闭 `torch.compile`，便于调试或兼容性 |
| `--fast_dev_run` | 快速跑几步即停，用于检查流程 |

示例（关闭编译、快速跑几步）：

```bash
uv run elefant/policy_model/train_future.py \
  --config config/policy_model/150M_local_dataset_future.yaml \
  --data_folder dataset \
  --no_compile \
  --fast_dev_run
```

---

## 5. 输出与恢复

- **Checkpoint 目录**：`{output_path}/stage3_future_vision/`  
  例如：`output/policy_model/150M_future/stage3_future_vision/`
- **保存规则**：按配置中的 `save_every_n_steps` 保存，文件名形如 `checkpoint-step=00010000.ckpt`。
- **恢复训练**：若该目录下已有 checkpoint，`train_stage3_future_vision` 会自动选择**最新**的 checkpoint 恢复；若需指定文件，可在配置中设置 `stage3_finetune.init.stage3_model_path` 指向具体 `.ckpt` 路径。

---

## 6. 日志与监控

- **WandB**：  
  - 项目名、entity 等由配置中 `wandb` 决定；  
  - 实验名会带后缀 `_stage3_future_vision`，例如 `150M-local-dataset-future_stage3_future_vision`。
- **训练指标**（每 50 步 log）：  
  - `train/loss_total`、`train/loss_action`、`train/loss_future_vision`  
  - `train/loss_key`、`train/loss_mouse_button`、`train/loss_mouse_delta_x`、`train/loss_mouse_delta_y`  
  验证阶段会计算并记录对应的 validation 指标。

---

## 7. 与基础 Stage3 的区别

| 项目 | 基础 Stage3（Stage3LabelledBCLightning） | Future 版（Stage3FutureVisionLightning） |
|------|------------------------------------------|------------------------------------------|
| 入口脚本 | `scripts/train_local_dataset.sh` → `train.py` | `scripts/train_local_dataset_future.sh` → `train_future.py` |
| 配置示例 | `150M_local_dataset.yaml` | `150M_local_dataset_future.yaml` |
| 模型 | PolicyCausalTransformer | PolicyFutureCausalTransformer（多 future_vision_head） |
| 损失 | 仅 action CE + z_loss 等 | action 损失 + future_vision_loss（MSE，权重 0.1） |
| 输出目录 | `output/policy_model/150M/stage3_finetune` | `output/policy_model/150M_future/stage3_future_vision` |

两者共用同一套数据格式与 `stage3_finetune.*` 配置结构；仅训练入口、模型类和输出路径不同。

---

## 8. 常见问题

- **显存不足**：在 `150M_local_dataset_future.yaml` 中把 `training_dataset.batch_size` 改为 1。
- **编译报错 / 运行异常**：可先加 `--no_compile` 确认是否为 `torch.compile` 导致。
- **未来帧监督用的特征**：当前实现使用**当前配置的 `image_tokenizer`** 对下一帧做 forward，再对空间维 mean 得到全局特征；与 DINOv2 全局特征思路一致。若使用 conv tokenizer，得到的是 conv 特征的 mean。
- **DINOv2 tokenizer**：目前支持尚不完备，若使用 DINOv2 需注意显存与编译相关限制；推荐以 **Conv** tokenizer 进行 Future 版训练与验证。

---

## 9. OpenP2P 标准实现：传给 Policy Model 的 token 序列（Conv vs DINOv2）

两种 tokenizer 的**接口一致**：都是 `image_tokenizer(img)` → `[B, T, n_img_tokens, embed_dim]`，再与 text / thinking / action_out / action 等拼成**同一结构的序列**，唯一差别是 **每帧的图像 token 个数** `n_img_tokens`。

### 序列构造方式（`policy_transformer.py`）

- 输入：`img [B, T, C, H, W]`、`action_embeddings_in [B, T, n_action_tokens, embed_dim]`、`text_tokens_embed [B, T, text_token_size, text_embed_dim]`。
- 对每一时间步 `t = 0..n_steps-1`，**单步序列**（在 `dim=1` 上拼接）为：
  ```text
  [ text_tokens | img_tokens | thinking_tokens | action_out_token | action_tokens ]
  ```
- 再在 `dim=1` 上把 `n_steps` 个这样的块按时间顺序拼成一条长序列 `x`，送入 transformer。
- 因此：
  ```text
  每步长度 = text_token_size + n_img_tokens + n_thinking_tokens + 1 + n_action_tokens
  max_seq_len = 每步长度 × n_steps
  ```

### 典型配置下的数值（`n_steps = 200`）

| 项 | 来源 | Conv 配置 | DINOv2 配置 |
|----|------|-----------|-------------|
| text_token_size | `text_embedding_shape[0]` | 1 | 1 |
| **n_img_tokens** | **image tokenizer** | **1** | **196** |
| n_thinking_tokens | 配置 | 1 | 1 |
| 1 | action_out 占位 | 1 | 1 |
| n_action_tokens | `action_mapping.get_seq_len() + 1`（如 Universal 4+2+2+1=9） | 9 | 9 |
| **每步长度** | 上五项之和 | **13** | **208** |
| **max_seq_len** | 每步长度 × 200 | **2 600** | **41 600** |

### Conv tokenizer 时

- `image_tokenizer(img)`：每帧 **1 个** 向量，整图一个全局表示。
- 传给 policy 的序列：每一步是 `[1 text | 1 img | 1 thinking | 1 action_out | 9 action]`，共 13 个 token/步；200 步 → 约 **2 600** 个 token。序列短，注意力矩阵小，不易 OOM。

### DINOv2 tokenizer 时

- `image_tokenizer(img)`：每帧 **196 个** patch tokens（ViT-B/14，192×192 → 14×14）。
- 传给 policy 的序列：每一步是 `[1 text | 196 img | 1 thinking | 1 action_out | 9 action]`，共 208 个 token/步；200 步 → 约 **41 600** 个 token。序列长，注意力矩阵大，必须用编译后的 flex_attention 才能避免物化全矩阵导致 OOM。

**结论**：OpenP2P 标准实现里，**结构上**两种 tokenizer 传给 policy 的都是「按步拼接的 text | img | thinking | action_out | action」；**数量上** Conv 每步 13 token、DINOv2 每步 208 token，总长差约 16 倍（2.6k vs 41.6k），因此显存与是否编译的敏感度完全不同。

---

以上即可完成 Future 版 Stage3（Stage3FutureVisionLightning）的训练与恢复；更多超参请直接编辑 `config/policy_model/150M_local_dataset_future.yaml`。
