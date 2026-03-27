# Inference 改造说明（Gamepad Native + 兼容回退）

本文档说明本仓库对 `elefant/policy_model/inference.py` 的改造目标、行为变化和联调方法。

---

## 1. 改造背景

当前 NitroGen 训练得到的是 **gamepad 动作空间**，但旧的推理传输协议（`video_inference.proto`）主要是键鼠结构。

在旧实现中，gamepad 推理结果会被编码进 `Action.keys`，例如：

- `gamepad:south`
- `gamepad:lx=0.1234`
- `gamepad:rt=0.7777`

这能保持兼容，但不是原生手柄协议，客户端需要额外解析字符串。

---

## 2. 本次改造目标

在不破坏现有链路的前提下支持路径 B（长期正确）：

- 优先走 **原生 gamepad 字段**（当 proto 已升级时）；
- 若 proto 尚未升级，则自动回退到旧的 `gamepad:*` keys 编码。

即：`native first, legacy fallback`。

---

## 3. 修改点（inference.py）

文件：`elefant/policy_model/inference.py`

新增/调整了以下逻辑：

1. `DecodedGamepadAction` 发送分支改为双栈：
   - 先尝试构造原生 gamepad `Action`
   - 失败则回退为 legacy keys 编码

2. 新增辅助方法：
   - `_build_native_gamepad_action(...)`
   - `_legacy_encode_gamepad_to_keys(...)`
   - `_empty_mouse_action()`
   - `_set_stick_fields(...)`

3. 原生字段识别为**运行时动态检查**（通过 protobuf descriptor）：
   - 尝试识别 `Action` 内的候选字段名：
     - `gamepad_action`
     - `game_pad_action`
     - `controller_action`
   - 若不存在，自动回退旧协议

---

## 4. 行为兼容性

| 场景 | 行为 |
|------|------|
| 旧 proto（无 gamepad 字段） | 继续发送 `keys = ["gamepad:..."]` |
| 新 proto（有 gamepad 字段） | 发送原生 gamepad 消息 |
| 键鼠模型推理 | 行为不变（keys + mouse_action） |

因此这次修改对现有键鼠链路无影响，对旧版客户端也保持可运行。

---

## 5. Recap 侧需要配合的内容

你们会在另一台主机修改 Recap。建议按以下最小集合改造：

1. 更新 Recap 所用 `video_inference.proto`，在 `Action` 中增加原生 gamepad 字段。
2. 客户端发送/接收端支持该字段的解析与序列化。
3. 执行层将原生 gamepad 动作注入虚拟手柄（Windows 建议 ViGEm/XInput 路径）。
4. 协议过渡期可保留对 `gamepad:*` keys 的解析（便于回滚/对比）。

---

## 6. 建议联调流程

1. **服务端启动**（当前仓库）：
   - 使用 gamepad 配置 + gamepad checkpoint 启动 `inference.py`
2. **检查日志/抓包**：
   - 升级前：应看到 legacy keys 编码
   - 升级后：应看到原生 gamepad 字段被填充
3. **Recap 执行验证**：
   - 验证按钮、左右摇杆、左右扳机都能正确注入
4. **回退验证**：
   - 临时使用旧 proto，确认 fallback 仍可工作

---

## 7. 已知限制

1. `use_full_inference` 目前对 gamepad 仍受限（仅 KV cache 路径可用）。
2. 设备选择逻辑仍偏向 `RTX 5090`，多卡 A100 环境建议显式控制可见设备（例如设置 `CUDA_VISIBLE_DEVICES=0`）。
3. 原生字段的最终命名应与 Recap 侧 proto 完全一致，否则会走回退分支。

---

## 8. 一句话结论

这次改造已经把服务端推理升级为“**原生 gamepad 优先 + 旧协议兜底**”。  
Recap 侧完成 proto 与执行适配后，即可无缝切到原生手柄链路。
