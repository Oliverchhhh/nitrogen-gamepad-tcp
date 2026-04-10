#!/bin/bash
# 使用本地 dataset 目录验证 Current（有 s0 / state target）Stage3 模型脚本

set -e  # 遇到错误立即退出

# 配置变量
CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_current.yaml"
DATA_FOLDER="NitroGen_cuphead_toy"
CHECKPOINT_PATH="output/policy_model/150M_nitrogen_current/stage3_future_vision/checkpoint-step=00030000.ckpt"
TEMP_CONFIG_FILE=""
MIN_STEPS=""
MAX_STEPS=""

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    echo "用法: bash scripts/validate_local_dataset_current.sh [-d 数据集路径] [-c 配置文件] [-k checkpoint路径] [--min_steps N] [--max_steps N]"
    echo
    echo "参数:"
    echo "  -d    数据集路径 (默认: NitroGen_cuphead_toy)"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_current.yaml)"
    echo "  -k    checkpoint 路径 (默认: output/policy_model/150M_nitrogen_current/stage3_future_vision/checkpoint-step=00030000.ckpt)"
    echo "  --min_steps   最小 step（可选，仅当 checkpoint_dir 是目录时有意义）"
    echo "  --max_steps   最大 step（可选，仅当 checkpoint_dir 是目录时有意义）"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/validate_local_dataset_current.sh"
    echo "  bash scripts/validate_local_dataset_current.sh -d NitroGen_cuphead_toy"
    echo "  bash scripts/validate_local_dataset_current.sh -d NitroGen_cuphead_toy -k output/policy_model/150M_nitrogen_current/stage3_future_vision/checkpoint-step=00030000.ckpt"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)
            DATA_FOLDER="$2"
            shift 2
            ;;
        -c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -k)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --min_steps)
            MIN_STEPS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        -h)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            usage
            exit 1
            ;;
    esac
done

cleanup() {
    if [ -n "$TEMP_CONFIG_FILE" ] && [ -f "$TEMP_CONFIG_FILE" ]; then
        rm -f "$TEMP_CONFIG_FILE"
    fi
}

trap cleanup EXIT

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Open P2P Current 验证脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查数据集路径
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

# 2. 检查配置文件和 checkpoint
echo -e "\n${YELLOW}[2/5] 检查配置文件和 checkpoint...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

if [ ! -e "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: checkpoint 不存在: $CHECKPOINT_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 配置文件存在${NC}"
echo -e "${GREEN}✓ checkpoint 存在${NC}"

# 2.5 生成临时配置文件，覆盖 local_prefix
TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_current_validate_config.XXXXXX.yaml)

if ! python - <<PY
from pathlib import Path
import yaml

config_path = Path("$CONFIG_FILE")
output_path = Path("$TEMP_CONFIG_FILE")
data_folder = "$DATA_FOLDER"

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if "stage3_finetune" not in cfg:
    raise SystemExit("配置文件缺少 stage3_finetune 字段")

if "training_dataset" in cfg["stage3_finetune"]:
    cfg["stage3_finetune"]["training_dataset"]["local_prefix"] = data_folder

if "validation_datasets" in cfg["stage3_finetune"]:
    for item in cfg["stage3_finetune"]["validation_datasets"]:
        item["local_prefix"] = data_folder

# 动态计算并覆盖 n_validation_steps，确保跑完整个验证集
# 原始配置 n_validation_steps=0 会导致只跑极少样本
import os, glob as _glob
val_datasets = cfg["stage3_finetune"].get("validation_datasets", [])
local_prefix = val_datasets[0]["local_prefix"] if val_datasets else data_folder
n_seq_timesteps = cfg.get("shared", {}).get("n_seq_timesteps", 200)
frames_per_chunk = 1200  # NitroGen 标准切片长度
protos = _glob.glob(os.path.join(local_prefix, "**/*.proto"), recursive=True)
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
    echo -e "${YELLOW}警告: 未检测到 nvidia-smi，可能没有 GPU${NC}"
fi

# 4. 设置环境变量和清理临时目录
echo -e "\n${YELLOW}[4/5] 设置环境变量和清理临时目录...${NC}"
unset HF_ENDPOINT

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_current_validation/torch_compiler/inductor_cache"
export CUDA_VISIBLE_DEVICES=0

echo -e "${GREEN}单卡模式: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

rm -rf /tmp/elefant_zmq /ephemeral/elefant_tmp_data /tmp/elefant_data 2>/dev/null || true
echo -e "${GREEN}✓ 环境准备完成${NC}"

# 5. 开始验证
echo -e "\n${YELLOW}[5/5] 开始验证 (Current + s0)...${NC}"
echo -e "${GREEN}配置文件: $TEMP_CONFIG_FILE${NC}"
echo -e "${GREEN}数据集路径: $DATA_FOLDER${NC}"
echo -e "${GREEN}checkpoint: $CHECKPOINT_PATH${NC}"
echo -e "${GREEN}========================================${NC}\n"

CMD=(python elefant/policy_model/validation.py
    --checkpoint_dir "$CHECKPOINT_PATH"
    --config_path "$TEMP_CONFIG_FILE"
)

if [ -n "$MIN_STEPS" ]; then
    CMD+=(--min_steps "$MIN_STEPS")
fi
if [ -n "$MAX_STEPS" ]; then
    CMD+=(--max_steps "$MAX_STEPS")
fi

"${CMD[@]}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}验证完成！${NC}"
echo -e "${GREEN}========================================${NC}"
