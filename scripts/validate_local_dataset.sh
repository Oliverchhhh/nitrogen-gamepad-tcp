#!/bin/bash
# 使用本地 dataset 目录验证 Stage3 模型脚本
# 支持 future/current 模式，支持键鼠/手柄，手柄模式下自动输出扩展指标
# 验证结果同时上报 wandb 并保存本地 JSON

set -e  # 遇到错误立即退出

# 配置变量
CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_current.yaml"
# CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset.yaml"
DATA_FOLDER="NitroGen_cuphead_toy"
# CHECKPOINT_PATH="/data2T/rjt/nitrogen-openp2p2-future-frame/output/policy_model/150M_nitrogen_cuphead_all/stage3_finetune/checkpoint-step=00100000.ckpt"
# CHECKPOINT_PATH="output_20260401/policy_model/150M_nitrogen/stage3_finetune/checkpoint-step=00040000.ckpt"
CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_all_current_gt_vjepa2/stage3_future_vision/checkpoint-step=00140000.ckpt"
TEMP_CONFIG_FILE=""
MIN_STEPS=""
MAX_STEPS=""

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    echo "用法: bash scripts/validate_local_dataset.sh [-d 数据集路径] [-c 配置文件] [-k checkpoint路径/目录] [--min_steps N] [--max_steps N]"
    echo
    echo "参数:"
    echo "  -d    验证数据集路径 (默认: NitroGen_cuphead_toy)"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_current.yaml)"
    echo "  -k    checkpoint 路径（单个 .ckpt 文件或包含多个 checkpoint 的目录）"
    echo "  --min_steps   最小 step（可选，仅当 -k 指向目录时有意义）"
    echo "  --max_steps   最大 step（可选，仅当 -k 指向目录时有意义）"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  # 验证单个 checkpoint"
    echo "  bash scripts/validate_local_dataset.sh -k output/policy_model/150M_nitrogen_current/stage3_future_vision/checkpoint-step=00040000.ckpt"
    echo
    echo "  # 验证目录下所有 checkpoint"
    echo "  bash scripts/validate_local_dataset.sh -k output/policy_model/150M_nitrogen_current/stage3_future_vision/"
    echo
    echo "  # 指定数据集和配置"
    echo "  bash scripts/validate_local_dataset.sh -d NitroGen_cuphead_toy -c config/policy_model/150M_local_nitrogen_dataset_current.yaml -k path/to/ckpt"
    echo
    echo "  # 只验证 step 30000~50000 的 checkpoint"
    echo "  bash scripts/validate_local_dataset.sh -k output/dir/ --min_steps 30000 --max_steps 50000"
    echo
    echo "输出:"
    echo "  - wandb: perplexity + 手柄扩展指标（button acc, stick MAE, direction acc 等）"
    echo "  - 本地 JSON: checkpoint 同目录下 validation_step{STEP}.json"
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
        -h|--help)
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

# 检查必须参数
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
echo -e "${GREEN}Open P2P 模型验证脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查数据集路径
echo -e "\n${YELLOW}[1/5] 检查数据集路径...${NC}"
if [ ! -d "$DATA_FOLDER" ]; then
    echo -e "${RED}错误: 数据集目录不存在: $DATA_FOLDER${NC}"
    echo -e "${YELLOW}提示: 可用 NitroGen_cuphead_toy 作为轻量验证集${NC}"
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

echo -e "${GREEN}✓ 配置文件: $CONFIG_FILE${NC}"
if [ -d "$CHECKPOINT_PATH" ]; then
    CKPT_COUNT=$(find "$CHECKPOINT_PATH" -name "*.ckpt" | wc -l)
    echo -e "${GREEN}✓ checkpoint 目录: $CHECKPOINT_PATH ($CKPT_COUNT 个 checkpoint)${NC}"
else
    echo -e "${GREEN}✓ checkpoint 文件: $CHECKPOINT_PATH${NC}"
fi

# 2.5 生成临时配置文件，覆盖 local_prefix 为验证数据集
TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_validate_config.XXXXXX.yaml)

if ! python3 - <<PY
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

# 离线验证需要跑完整个数据集，强制覆盖 n_validation_steps
# 原始配置可能设为 0（训练期间跳过）
# ignore_iterator_reset=True 时 dataloader 会无限循环，必须用有限步数控制
# 正确计算：n_protos × (帧数 / n_seq_timesteps) / batch_size = 独立样本数
# NitroGen_cuphead_toy: 12 proto × (1200 / 200) / 1 = 72 steps（跑完一个 epoch）
import os, glob
val_datasets = cfg["stage3_finetune"].get("validation_datasets", [])
local_prefix = val_datasets[0]["local_prefix"] if val_datasets else data_folder
n_seq_timesteps = cfg.get("shared", {}).get("n_seq_timesteps", 200)
frames_per_chunk = 1200  # NitroGen 标准切片长度
protos = glob.glob(os.path.join(local_prefix, "**/*.proto"), recursive=True)
n_protos = len(protos)
# BATCH_SIZE_FOR_VAL=1，每个 proto 产出 frames_per_chunk // n_seq_timesteps 个样本
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
echo -e "\n${YELLOW}[4/5] 设置环境变量...${NC}"
unset HF_ENDPOINT

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_validation/torch_compiler/inductor_cache"
export CUDA_VISIBLE_DEVICES=0

echo -e "${GREEN}单卡模式: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

rm -rf /tmp/elefant_zmq /ephemeral/elefant_tmp_data /tmp/elefant_data 2>/dev/null || true
echo -e "${GREEN}✓ 环境准备完成${NC}"

# 5. 开始验证
echo -e "\n${YELLOW}[5/5] 开始验证...${NC}"
echo -e "${GREEN}配置文件: $CONFIG_FILE (临时: $TEMP_CONFIG_FILE)${NC}"
echo -e "${GREEN}数据集路径: $DATA_FOLDER${NC}"
echo -e "${GREEN}checkpoint: $CHECKPOINT_PATH${NC}"
if [ -n "$MIN_STEPS" ]; then
    echo -e "${GREEN}min_steps: $MIN_STEPS${NC}"
fi
if [ -n "$MAX_STEPS" ]; then
    echo -e "${GREEN}max_steps: $MAX_STEPS${NC}"
fi
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}验证指标:${NC}"
echo -e "${YELLOW}  - Perplexity (所有模式)${NC}"
echo -e "${YELLOW}  - 手柄扩展指标 (手柄模式自动启用):${NC}"
echo -e "${YELLOW}      Button: accuracy, exact_match, F1${NC}"
echo -e "${YELLOW}      Stick:  MAE, RMSE, direction_acc, deadzone_acc${NC}"
echo -e "${YELLOW}      Trigger: MAE, binary_acc${NC}"
echo -e "${YELLOW}  - 输出: wandb + 本地 JSON${NC}"
echo -e "${GREEN}========================================${NC}\n"

CMD=(python3 elefant/policy_model/validation.py
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
echo -e "${GREEN}结果已保存:${NC}"
echo -e "${GREEN}  - wandb: 查看 project dashboard${NC}"
echo -e "${GREEN}  - 本地 JSON: checkpoint 同目录下 validation_step*.json${NC}"
echo -e "${GREEN}========================================${NC}"
