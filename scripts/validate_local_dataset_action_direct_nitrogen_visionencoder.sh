#!/bin/bash
# Teacher-forcing 验证脚本（nitrogen_visionencoder direct action）
# 对应评估代码: elefant/policy_model/validation_action_direct.py

set -e

LOG_DIR="val_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/validate_action_direct_nitrogen_visionencoder_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action_direct_nitrogen_visionencoder.yaml"
# DATA_FOLDER="cuphead_dataset_converted/v350335326"
DATA_FOLDER="cuphead_one_level_3"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_nitrogen_visionencoder_F18_zero_action"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_nitrogen_visionencoder_F18_zero_action"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_nitrogen_visionencoder_F18_zero_action_test"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_nitrogen_visionencoder_F18_zero_action_v350335326"
CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_nitrogen_visionencoder_F18_zero_action_all"
TEMP_CONFIG_FILE=""
MIN_STEPS=""
MAX_STEPS=""
N_SEQUENCES="300"
GPU_ID="2"
NO_WANDB=true
FRAME_INDEX="0"
ALL_FUTURE_FRAMES=false

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/validate_local_dataset_action_direct_nitrogen_visionencoder.sh [-d 数据集] [-c 配置] [-k checkpoint路径/目录] [-g GPU_ID] [--no_wandb] [--min_steps N] [--max_steps N] [--frame_index N] [--all_future_frames]"
    echo
    echo "参数:"
    echo "  -d    验证数据集路径 (默认: cuphead_one_level_1)"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action_direct_nitrogen_visionencoder.yaml)"
    echo "  -k    checkpoint 路径（支持 .ckpt 文件、stage3_finetune 目录、或实验根目录）"
    echo "  -g    GPU ID (默认: 0)"
    echo "  -n    限制验证的轨迹序列数量（可选，默认跑完整个验证集）"
    echo "  --no_wandb    禁用 wandb 上报，仅保存本地 JSON（默认开启）"
    echo "  --min_steps   最小 step（可选，仅当 -k 指向目录时有意义）"
    echo "  --max_steps   最大 step（可选，仅当 -k 指向目录时有意义）"
    echo "  --frame_index 仅评估指定 future frame（默认: 0）"
    echo "  --all_future_frames 评估全部 future frames 并输出逐帧+汇总指标"
    echo "  -h    显示帮助"
}

resolve_checkpoint_path() {
    local raw_path="$1"

    if [ -f "$raw_path" ] && [[ "$raw_path" == *.ckpt ]]; then
        echo "$raw_path"
        return 0
    fi

    if [ -d "$raw_path" ]; then
        if compgen -G "$raw_path/*.ckpt" > /dev/null; then
            echo "$raw_path"
            return 0
        fi
        if [ -d "$raw_path/stage3_finetune" ] && compgen -G "$raw_path/stage3_finetune/*.ckpt" > /dev/null; then
            echo "$raw_path/stage3_finetune"
            return 0
        fi
    fi

    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) DATA_FOLDER="$2"; shift 2 ;;
        -c) CONFIG_FILE="$2"; shift 2 ;;
        -k) CHECKPOINT_PATH="$2"; shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        -n) N_SEQUENCES="$2"; shift 2 ;;
        --no_wandb) NO_WANDB=true; shift ;;
        --min_steps) MIN_STEPS="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --frame_index) FRAME_INDEX="$2"; shift 2 ;;
        --all_future_frames) ALL_FUTURE_FRAMES=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo -e "${RED}错误: 未知参数 $1${NC}"; usage; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: 必须通过 -k 指定 checkpoint 路径或目录${NC}"
    usage
    exit 1
fi

if ! RESOLVED_CHECKPOINT_PATH=$(resolve_checkpoint_path "$CHECKPOINT_PATH"); then
    echo -e "${RED}错误: checkpoint 路径无效或未找到 .ckpt: $CHECKPOINT_PATH${NC}"
    echo -e "${YELLOW}提示: 可传入 .ckpt 文件、stage3_finetune 目录、或实验根目录${NC}"
    exit 1
fi
CHECKPOINT_PATH="$RESOLVED_CHECKPOINT_PATH"

cleanup() {
    if [ -n "$TEMP_CONFIG_FILE" ] && [ -f "$TEMP_CONFIG_FILE" ]; then
        rm -f "$TEMP_CONFIG_FILE"
    fi
}
trap cleanup EXIT

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Nitrogen VisionEncoder Direct Action 验证脚本${NC}"
echo -e "${GREEN}========================================${NC}"

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

echo -e "\n${YELLOW}[2/5] 检查配置文件和 checkpoint...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 配置文件: $CONFIG_FILE${NC}"
if [ -d "$CHECKPOINT_PATH" ]; then
    CKPT_COUNT=$(find "$CHECKPOINT_PATH" -maxdepth 1 -name "*.ckpt" | wc -l)
    echo -e "${GREEN}✓ checkpoint 目录: $CHECKPOINT_PATH ($CKPT_COUNT 个 checkpoint)${NC}"
else
    echo -e "${GREEN}✓ checkpoint 文件: $CHECKPOINT_PATH${NC}"
fi

echo -e "\n${YELLOW}[2.5/5] 生成临时验证配置...${NC}"
TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_validate_direct_visionencoder_config.XXXXXX.yaml)

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
if "validation_datasets" not in stage3 or not stage3.get("validation_datasets"):
    single_val = stage3.get("validation_dataset")
    if single_val is not None:
        stage3["validation_datasets"] = [single_val]
    elif "training_dataset" in stage3:
        fallback_val = dict(stage3["training_dataset"])
        fallback_val["validation_name"] = "validation_default"
        stage3["validation_datasets"] = [fallback_val]
    else:
        raise SystemExit("配置缺少 validation_datasets/validation_dataset/training_dataset，无法构建验证集")

for i, item in enumerate(stage3["validation_datasets"]):
    item["local_prefix"] = data_folder
    if "validation_name" not in item or item["validation_name"] in (None, ""):
        item["validation_name"] = f"validation_{i}"

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
echo -e "${GREEN}✓ 临时配置文件: $TEMP_CONFIG_FILE${NC}"

echo -e "\n${YELLOW}[3/5] 检查 GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU 检查完成${NC}"
else
    echo -e "${YELLOW}警告: 未检测到 nvidia-smi${NC}"
fi

echo -e "\n${YELLOW}[4/5] 设置环境变量...${NC}"
unset HF_ENDPOINT
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_validation_direct_visionencoder/torch_compiler/inductor_cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
rm -rf /tmp/elefant_zmq /ephemeral/elefant_tmp_data /tmp/elefant_data 2>/dev/null || true
echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

echo -e "\n${YELLOW}[5/5] 开始 teacher-forcing 验证...${NC}"
echo -e "${GREEN}配置: $CONFIG_FILE${NC}"
echo -e "${GREEN}数据集: $DATA_FOLDER${NC}"
echo -e "${GREEN}checkpoint: $CHECKPOINT_PATH${NC}"
[ -n "$MIN_STEPS" ] && echo -e "${GREEN}min_steps: $MIN_STEPS${NC}"
[ -n "$MAX_STEPS" ] && echo -e "${GREEN}max_steps: $MAX_STEPS${NC}"
[ -n "$N_SEQUENCES" ] && echo -e "${GREEN}n_sequences (限制): $N_SEQUENCES${NC}"
if [ "$ALL_FUTURE_FRAMES" = true ]; then
    echo -e "${GREEN}评估模式: all_future_frames${NC}"
else
    echo -e "${GREEN}评估模式: frame_index=${FRAME_INDEX}${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

CMD=(python3 elefant/policy_model/validation_action_direct.py
    --checkpoint_dir "$CHECKPOINT_PATH"
    --config_path "$TEMP_CONFIG_FILE"
    --data_folder "$DATA_FOLDER"
)
[ -n "$MIN_STEPS" ] && CMD+=(--min_steps "$MIN_STEPS")
[ -n "$MAX_STEPS" ] && CMD+=(--max_steps "$MAX_STEPS")
[ -n "$N_SEQUENCES" ] && CMD+=(--n_sequences "$N_SEQUENCES")
[ "$NO_WANDB" = true ] && CMD+=(--no_wandb)
[ "$ALL_FUTURE_FRAMES" = true ] && CMD+=(--all_future_frames)
[ "$ALL_FUTURE_FRAMES" != true ] && CMD+=(--frame_index "$FRAME_INDEX")

"${CMD[@]}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Teacher-forcing 验证完成！${NC}"
echo -e "${GREEN}结果: checkpoint 同目录下 validation_direct_step*.json${NC}"
echo -e "${GREEN}========================================${NC}"
