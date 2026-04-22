#!/bin/bash
# Teacher-forcing 验证脚本 — direct action 范式
# 对应评估代码: elefant/policy_model/validation_action_direct.py

set -e

CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml"
DATA_FOLDER="NitroGen_cuphead_toy"
CHECKPOINT_PATH="/data2T/rjt/nitrogen-openp2p2-future-frame/output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00130000.ckpt"
TEMP_CONFIG_FILE=""
MIN_STEPS=""
MAX_STEPS=""
GPU_ID="0"
NO_WANDB=true

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/validate_local_dataset_action_direct.sh [-d 数据集] [-c 配置] [-k checkpoint路径/目录] [-g GPU_ID] [--no_wandb] [--min_steps N] [--max_steps N]"
    echo
    echo "参数:"
    echo "  -d    验证数据集路径 (默认: cuphead_dataset_converted)"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml)"
    echo "  -k    checkpoint 路径（单个 .ckpt 文件或包含多个 checkpoint 的目录）"
    echo "  -g    GPU ID (默认: 0)"
    echo "  --no_wandb    禁用 wandb 上报，仅保存本地 JSON"
    echo "  --min_steps   最小 step（可选，仅当 -k 指向目录时有意义）"
    echo "  --max_steps   最大 step（可选，仅当 -k 指向目录时有意义）"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/validate_local_dataset_action_direct.sh \\"
    echo "    -k output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_all/stage3_finetune/checkpoint-step=00040000.ckpt"
    echo
    echo "  bash scripts/validate_local_dataset_action_direct.sh \\"
    echo "    -k output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_all/stage3_finetune/ --min_steps 30000 --max_steps 50000"
    echo
    echo "输出:"
    echo "  - wandb: button acc/F1, stick MAE/RMSE/direction_acc/deadzone_acc + perplexity"
    echo "  - 本地 JSON: checkpoint 同目录下 validation_direct_step{STEP}.json"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) DATA_FOLDER="$2"; shift 2 ;;
        -c) CONFIG_FILE="$2"; shift 2 ;;
        -k) CHECKPOINT_PATH="$2"; shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        --no_wandb) NO_WANDB=true; shift ;;
        --min_steps) MIN_STEPS="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo -e "${RED}错误: 未知参数 $1${NC}"; usage; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: 必须通过 -k 指定 checkpoint 路径或目录${NC}"
    usage
    exit 1
fi

cleanup() {
    if [ -n "$TEMP_CONFIG_FILE" ] && [ -f "$TEMP_CONFIG_FILE" ]; then
        rm -f "$TEMP_CONFIG_FILE"
    fi
}
trap cleanup EXIT

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Direct Action Teacher-Forcing 验证脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查数据集
echo -e "\n${YELLOW}[1/5] 检查数据集路径...${NC}"
if [ ! -d "$DATA_FOLDER" ]; then
    echo -e "${RED}错误: 数据集目录不存在: $DATA_FOLDER${NC}"
    exit 1
fi
PROTO_COUNT=$(find -L "$DATA_FOLDER" -name "*.proto" | wc -l)
if [ "$PROTO_COUNT" -eq 0 ]; then
    echo -e "${RED}错误: 在 $DATA_FOLDER 中未找到 .proto 文件${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 找到 $PROTO_COUNT 个标注文件${NC}"

# 2. 检查配置和 checkpoint
echo -e "\n${YELLOW}[2/5] 检查配置文件和 checkpoint...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi
if [ ! -e "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: checkpoint 不存在: $CHECKPOINT_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 配置文件: $CONFIG_FILE${NC}"
if [ -d "$CHECKPOINT_PATH" ]; then
    CKPT_COUNT=$(find "$CHECKPOINT_PATH" -name "*.ckpt" | wc -l)
    echo -e "${GREEN}✓ checkpoint 目录: $CHECKPOINT_PATH ($CKPT_COUNT 个 checkpoint)${NC}"
else
    echo -e "${GREEN}✓ checkpoint 文件: $CHECKPOINT_PATH${NC}"
fi

# 2.5 生成临时配置（覆盖 local_prefix 和 n_validation_steps）
TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_validate_direct_config.XXXXXX.yaml)

if ! python3 - <<PY
from pathlib import Path
import yaml, os, glob

config_path = Path("$CONFIG_FILE")
output_path = Path("$TEMP_CONFIG_FILE")
data_folder = "$DATA_FOLDER"

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if "stage3_finetune" not in cfg:
    raise SystemExit("配置文件缺少 stage3_finetune 字段")

if "training_dataset" in cfg["stage3_finetune"]:
    cfg["stage3_finetune"]["training_dataset"]["local_prefix"] = data_folder

stage3 = cfg["stage3_finetune"]

# 兼容两种写法：
# 1) validation_datasets: [...]
# 2) validation_dataset: {...}
if "validation_datasets" not in stage3 or not stage3.get("validation_datasets"):
    single_val = stage3.get("validation_dataset")
    if single_val is not None:
        stage3["validation_datasets"] = [single_val]
    elif "training_dataset" in stage3:
        # 若未显式提供验证集，回退到训练集路径（离线评估常见）
        fallback_val = dict(stage3["training_dataset"])
        fallback_val["validation_name"] = "validation_default"
        stage3["validation_datasets"] = [fallback_val]
    else:
        raise SystemExit("配置缺少 validation_datasets/validation_dataset/training_dataset，无法构建验证集")

for i, item in enumerate(stage3["validation_datasets"]):
    item["local_prefix"] = data_folder
    # ValidationDatasetConfig 要求必填 validation_name
    if "validation_name" not in item or item["validation_name"] in (None, ""):
        item["validation_name"] = f"validation_{i}"

# 自动计算 n_validation_steps（跑完整个验证集一个 epoch）
val_datasets = stage3.get("validation_datasets", [])
if not val_datasets:
    raise SystemExit("validation_datasets 为空，无法进行离线验证")
local_prefix = val_datasets[0]["local_prefix"] if val_datasets else data_folder
n_seq_timesteps = cfg.get("shared", {}).get("n_seq_timesteps", 200)
frames_per_chunk = 1200
protos = glob.glob(os.path.join(local_prefix, "**/*.proto"), recursive=True)
n_protos = len(protos)
n_validation_steps = n_protos * (frames_per_chunk // n_seq_timesteps)
cfg["stage3_finetune"]["n_validation_steps"] = max(1, n_validation_steps)

with output_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
then
    echo -e "${RED}错误: 生成临时验证配置失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 已生成临时配置文件: $TEMP_CONFIG_FILE${NC}"

# 3. 检查 GPU
echo -e "\n${YELLOW}[3/5] 检查 GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU 检查完成${NC}"
else
    echo -e "${YELLOW}警告: 未检测到 nvidia-smi${NC}"
fi

# 4. 环境变量
echo -e "\n${YELLOW}[4/5] 设置环境变量...${NC}"
unset HF_ENDPOINT
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_validation_direct/torch_compiler/inductor_cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
rm -rf /tmp/elefant_zmq /ephemeral/elefant_tmp_data /tmp/elefant_data 2>/dev/null || true
echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

# 5. 运行验证
echo -e "\n${YELLOW}[5/5] 开始 teacher-forcing 验证...${NC}"
echo -e "${GREEN}配置: $CONFIG_FILE  数据集: $DATA_FOLDER  checkpoint: $CHECKPOINT_PATH${NC}"
[ -n "$MIN_STEPS" ] && echo -e "${GREEN}min_steps: $MIN_STEPS${NC}"
[ -n "$MAX_STEPS" ] && echo -e "${GREEN}max_steps: $MAX_STEPS${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}验证指标:${NC}"
echo -e "${YELLOW}  - Perplexity (button BCE + stick CE)${NC}"
echo -e "${YELLOW}  - Button: accuracy, exact_match, F1${NC}"
echo -e "${YELLOW}  - Stick:  MAE, RMSE, direction_acc, deadzone_acc${NC}"
echo -e "${YELLOW}  - 输出: 本地 JSON (validation_direct_step*.json)  [wandb 默认禁用，传 --no_wandb=false 可开启]${NC}"
echo -e "${GREEN}========================================${NC}\n"

CMD=(python3 elefant/policy_model/validation_action_direct.py
    --checkpoint_dir "$CHECKPOINT_PATH"
    --config_path "$TEMP_CONFIG_FILE"
)
[ -n "$MIN_STEPS" ] && CMD+=(--min_steps "$MIN_STEPS")
[ -n "$MAX_STEPS" ] && CMD+=(--max_steps "$MAX_STEPS")
[ "$NO_WANDB" = true ] && CMD+=(--no_wandb)

"${CMD[@]}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Teacher-forcing 验证完成！${NC}"
echo -e "${GREEN}结果: checkpoint 同目录下 validation_direct_step*.json${NC}"
echo -e "${GREEN}========================================${NC}"
