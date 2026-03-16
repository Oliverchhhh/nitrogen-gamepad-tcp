# Gamepad on NitroGen 训练说明

本文档说明如何在当前仓库中，使用 **NitroGen 数据**训练 `open-p2p` 的 **gamepad action mapping** 分支。

---

## 1. 当前实现范围

- 数据侧：支持 NitroGen `actions_processed.parquet` 转换为 `annotation.proto`
- 动作空间：支持 `gamepad`（按钮 + 双摇杆 + 双扳机）
- 训练侧：`stage3_finetune` 已支持 gamepad 标签与损失
- 推理侧：支持 `gamepad` mapping（当前协议兼容输出，见下文注意事项）

---

## 2. 数据准备流程

### 2.1 将 NitroGen chunk 转为 open-p2p 样式

可使用脚本：

- `scripts/convert_nitrogen_to_openp2p.py`

单个 chunk 示例：

```bash
python scripts/convert_nitrogen_to_openp2p.py \
  --input /mnt/d/nitrogen_samples_100/cuphead/lable_data/actions/SHARD_0045/v350335326/v350335326_chunk_0011 \
  --output /mnt/d/project/open-p2p-main/dataset_nitrogen_toy \
  --copy-video \
  --generate-192 \
  --overwrite
```

批量示例：

```bash
python scripts/convert_nitrogen_to_openp2p.py \
  --input /mnt/d/nitrogen_samples_100/cuphead/lable_data/actions \
  --output /mnt/d/project/open-p2p-main/dataset_nitrogen_toy \
  --recursive \
  --copy-video \
  --generate-192 \
  --video-root /mnt/d/nitrogen_samples_100 \
  --overwrite
```

转换后每个样本目录应包含：

- `annotation.proto`
- `video.mp4`
- `192x192.mp4`

---

## 3. 配置文件

推荐使用：

- `config/policy_model/150M_local_nitrogen_dataset.yaml`

关键字段：

- `shared.action_mapping_type: "gamepad"`
- `shared.gamepad_action_mapping.*`
- `stage3_finetune.training_dataset.local_prefix: "dataset_nitrogen_toy"`
- `stage3_finetune.validation_datasets[].local_prefix: "dataset_nitrogen_toy"`

---

## 4. 训练命令（推荐 no_compile）

由于当前环境下 `torch.compile` 可能触发 Triton/Inductor 编译错误，建议先使用 no_compile 脚本：

```bash
bash scripts/train_local_dataset_no_compile.sh \
  -c config/policy_model/150M_local_nitrogen_dataset.yaml \
  -d dataset_nitrogen_toy \
  -o output_nitrogen
```

该脚本会调用：

```bash
uv run elefant/policy_model/train.py --config <config> --data_folder <dir> --no_compile
```

---

## 5. 快速验证是否跑在 gamepad 分支

训练启动后，检查日志中是否出现：

- `action_mapping_type='gamepad'`
- 模型 summary 中存在：
  - `gamepad_button_embedding`
  - `gamepad_stick_embedding`
  - `gamepad_trigger_embedding`
  - `gamepad_*_out_logits`

如果仍看到 `keyboard_out_logits / mouse_*`，说明没有进入 gamepad 分支。

---

## 6. 常见问题

### Q1: `Tensor has no attribute 'keys'`

说明训练仍走键鼠标签路径，而 batch 是 gamepad tensor 标签。请确认：

- 代码包含 gamepad 训练分支改动
- 配置里 `shared.action_mapping_type` 为 `gamepad`

### Q2: `torch._inductor.exc.InductorError` / `PassManager::run failed`

这是 `torch.compile` 后端编译问题，不是数据格式问题。建议：

- 先用 `--no_compile` 跑通
- 再评估保守 compile 策略

---

## 7. 当前推理协议注意事项

当前推理通路已支持 gamepad mapping，但 `video_inference.proto` 仍是键鼠结构。  
因此 gamepad 推理结果暂以兼容形式编码在 `Action.keys`（`gamepad:*` 前缀）中返回。

若需要原生 gamepad 推理协议，需扩展 proto 并同步 Recap/客户端解析。

