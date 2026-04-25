#!/bin/bash
# Per-offset 验证脚本 — direct action 范式
# 对应评估代码: elefant/policy_model/validation_per_offset_action_direct.py
#
# 验证思路：每帧 F 个 ain tokens，a_in^f 预测 t+f 帧动作，
# 分别计算 offset=0..F-1 的性能指标，输出衰减曲线。

set -e

CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml"
DATA_FOLDER="NitroGen_cuphead_toy"
CHECKPOINT_PATH="/data2T/rjt/nitrogen-openp2p2-future-frame/output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00130000.ckpt"
MIN_STEPS=""
MAX_STEPS=""
N_VALIDATION_STEPS="18"
GPU_ID="0"
NO_WANDB=true

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/validate_local_dataset_per_offset_action_direct.sh [-d 数据集] [-c 配置] [-k checkpoint路径/目录] [-g GPU_ID] [-n N] [--no_wandb] [--min_steps N] [--max_steps N]"
    echo
    echo "参数:"
    echo "  -d    验证数据集路径 (默认: NitroGen_cuphead_toy)"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml)"
    echo "  -k    checkpoint 路径（单个 .ckpt 文件或包含多个 checkpoint 的目录）"
    echo "  -g    GPU ID (默认: 0)"
    echo "  -n    评估的 batch 数量（建议显式指定，如 200）；不传则沿用 config 中的"
    echo "        n_validation_steps，若 config 该值为 0 则实际只跑 1 个 batch"
    echo "  --no_wandb    禁用 wandb 上报，仅保存本地 JSON"
    echo "  --min_steps   最小 step（可选，仅当 -k 指向目录时有意义）"
    echo "  --max_steps   最大 step（可选，仅当 -k 指向目录时有意义）"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/validate_local_dataset_per_offset_action_direct.sh \\"
    echo "    -k output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00130000.ckpt"
    echo
    echo "  bash scripts/validate_local_dataset_per_offset_action_direct.sh \\"
    echo "    -k output/policy_model/.../stage3_finetune/ --min_steps 100000 --max_steps 130000"
    echo
    echo "输出:"
    echo "  - 本地 JSON: checkpoint 同目录下 validation_per_offset_step{STEP}.json"
    echo "  - 包含 offset_00 ~ offset_{F-1} 各自的 button/stick 指标及 summary 衰减曲线"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) DATA_FOLDER="$2"; shift 2 ;;
        -c) CONFIG_FILE="$2"; shift 2 ;;
        -k) CHECKPOINT_PATH="$2"; shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        -n) N_VALIDATION_STEPS="$2"; shift 2 ;;
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

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Per-Offset Direct Action 验证脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查数据集
echo -e "\n${YELLOW}[1/4] 检查数据集路径...${NC}"
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
echo -e "\n${YELLOW}[2/4] 检查配置文件和 checkpoint...${NC}"
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

# 3. 检查 GPU 和设置环境变量
echo -e "\n${YELLOW}[3/4] 检查 GPU 并设置环境变量...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU 检查完成${NC}"
else
    echo -e "${YELLOW}警告: 未检测到 nvidia-smi${NC}"
fi
unset HF_ENDPOINT
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_validation_per_offset/torch_compiler/inductor_cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
rm -rf /tmp/elefant_zmq /ephemeral/elefant_tmp_data /tmp/elefant_data 2>/dev/null || true
echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

# 4. 运行验证
echo -e "\n${YELLOW}[4/4] 开始 per-offset 验证...${NC}"
echo -e "${GREEN}配置: $CONFIG_FILE  数据集: $DATA_FOLDER  checkpoint: $CHECKPOINT_PATH${NC}"
[ -n "$MIN_STEPS" ]          && echo -e "${GREEN}min_steps: $MIN_STEPS${NC}"
[ -n "$MAX_STEPS" ]          && echo -e "${GREEN}max_steps: $MAX_STEPS${NC}"
[ -n "$N_VALIDATION_STEPS" ] && echo -e "${GREEN}n_validation_steps: $N_VALIDATION_STEPS${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}验证指标 (每个 offset 独立统计):${NC}"
echo -e "${YELLOW}  - Button: accuracy, exact_match, F1${NC}"
echo -e "${YELLOW}  - Stick:  MAE, RMSE, direction_acc, deadzone_acc${NC}"
echo -e "${YELLOW}  - 输出: 本地 JSON (validation_per_offset_step*.json)${NC}"
echo -e "${GREEN}========================================${NC}\n"

CMD=(python3 elefant/policy_model/validation_per_offset_action_direct.py
    --checkpoint_dir "$CHECKPOINT_PATH"
    --config_path    "$CONFIG_FILE"
    --data_folder    "$DATA_FOLDER"
)
[ -n "$MIN_STEPS" ]          && CMD+=(--min_steps "$MIN_STEPS")
[ -n "$MAX_STEPS" ]          && CMD+=(--max_steps "$MAX_STEPS")
[ -n "$N_VALIDATION_STEPS" ] && CMD+=(--n_validation_steps "$N_VALIDATION_STEPS")
[ "$NO_WANDB" = true ]       && CMD+=(--no_wandb)

"${CMD[@]}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Per-offset 验证完成！${NC}"
echo -e "${GREEN}结果: checkpoint 同目录下 validation_per_offset_step*.json${NC}"
echo -e "${GREEN}========================================${NC}"
