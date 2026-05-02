#!/bin/bash
# 自回归验证脚本 — direct action 范式
# 逐帧用模型自己采样的 action 作为下一步输入，贴近真实推理效果
# 对应评估代码: elefant/policy_model/validation_autoregressive_action_direct.py

set -e

LOG_DIR="val_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/validate_ar_action_direct_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

CONFIG_FILE="config/policy_model/600M_local_nitrogen_dataset_future_action_direct.yaml"
# CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action_direct_ckpt00280000_compat.yaml"
DATA_FOLDER="NitroGen_cuphead_toy"
# DATA_FOLDER="cuphead_one_level_3"
# CHECKPOINT_PATH="output_20260420/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00100000.ckpt"
#CHECKPOINT_PATH="output_20260420/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_onelevel_3/stage3_finetune/checkpoint-step=00030000.ckpt"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_v350335326/stage3_finetune/checkpoint-step=00080000.ckpt"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_onelevel3/stage3_finetune/checkpoint-step=00030000.ckpt"
#CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_v350335326/stage3_finetune/checkpoint-step=00150000.ckpt"
#CHECKPOINT_PATH="output_20260425/policy_model/150M_nitrogen_cuphead_future_action_direct_F1_2head_action_all/stage3_finetune/checkpoint-step=00260000.ckpt"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_v350335326/stage3_finetune/checkpoint-step=00100000.ckpt"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00300000.ckpt"
CHECKPOINT_PATH="output/policy_model/600M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_all/stage3_finetune/checkpoint-step=00050000.ckpt"
N_SEQUENCES=72
GPU_ID="0"
FULL_CAUSAL_MASK=False

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/validate_local_dataset_autoregressive_action_direct.sh -k checkpoint路径 [-d 数据集] [-c 配置] [-n 序列数] [-g GPU_ID] [-M]"
    echo
    echo "参数:"
    echo "  -k    checkpoint 路径（必填，单个 .ckpt 文件）"
    echo "  -d    验证数据集路径 (默认: cuphead_dataset_converted)"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml)"
    echo "  -n    验证的视频序列数量 (默认: 72)"
    echo "  -g    GPU ID (默认: 0)"
    echo "  -E    启用 torch.compile 模式"
    echo "  -C    禁用 torch.compile，退回 eager 模式（排查数值偏差用）"
    echo "  -M    使用完整 causal mask（禁用 block mask，用于 AR 对照实验）"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/validate_local_dataset_autoregressive_action_direct.sh \\"
    echo "    -k output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_all/stage3_finetune/checkpoint-step=00040000.ckpt"
    echo
    echo "  bash scripts/validate_local_dataset_autoregressive_action_direct.sh \\"
    echo "    -k path/to/ckpt -d cuphead_dataset_converted -n 20"
    echo
    echo "输出:"
    echo "  - 本地 JSON: checkpoint 同目录下 validation_ar_direct_step{STEP}.json"
}

while getopts ":k:d:c:n:g:ECMh" opt; do
    case "$opt" in
        k) CHECKPOINT_PATH="$OPTARG" ;;
        d) DATA_FOLDER="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        n) N_SEQUENCES="$OPTARG" ;;
        g) GPU_ID="$OPTARG" ;;
        E) TORCH_COMPILE_DISABLE=0 ;;
        C) TORCH_COMPILE_DISABLE=1 ;;
        M) FULL_CAUSAL_MASK=true ;;
        h) usage; exit 0 ;;
        \?) echo -e "${RED}错误: 未知参数 -$OPTARG${NC}"; usage; exit 1 ;;
        :) echo -e "${RED}错误: 参数 -$OPTARG 缺少值${NC}"; usage; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: 必须通过 -k 指定 checkpoint 路径${NC}"
    usage
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Direct Action 自回归验证脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查输入
echo -e "\n${YELLOW}[1/3] 检查输入...${NC}"
[ ! -d "$DATA_FOLDER" ] && echo -e "${RED}数据集不存在: $DATA_FOLDER${NC}" && exit 1
[ ! -f "$CONFIG_FILE" ] && echo -e "${RED}配置文件不存在: $CONFIG_FILE${NC}" && exit 1
[ ! -f "$CHECKPOINT_PATH" ] && echo -e "${RED}checkpoint 不存在: $CHECKPOINT_PATH${NC}" && exit 1

PROTO_COUNT=$(find -L "$DATA_FOLDER" -name "*.proto" | wc -l)
echo -e "${GREEN}✓ 数据集: $DATA_FOLDER ($PROTO_COUNT 个标注)${NC}"
echo -e "${GREEN}✓ 配置: $CONFIG_FILE${NC}"
echo -e "${GREEN}✓ checkpoint: $CHECKPOINT_PATH${NC}"
echo -e "${GREEN}✓ 序列数: $N_SEQUENCES${NC}"

# 2. 环境
echo -e "\n${YELLOW}[2/3] 设置环境...${NC}"
unset HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_ar_direct_validation/torch_compiler/inductor_cache"
# 设为 0 启用 torch.compile；设为 1 禁用并退回 eager 模式
# 可通过 -E / -C 显式覆盖；未指定时默认禁用（1）
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"
echo -e "${GREEN}✓ compile 缓存: ${TORCHINDUCTOR_CACHE_DIR}${NC}"
echo -e "${GREEN}✓ TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE}${NC}"
echo -e "${GREEN}✓ FULL_CAUSAL_MASK=${FULL_CAUSAL_MASK}${NC}"

# 3. 运行
echo -e "\n${YELLOW}[3/3] 开始自回归验证...${NC}"
echo -e "${GREEN}序列数: $N_SEQUENCES${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}验证指标:${NC}"
echo -e "${YELLOW}  - Button: accuracy, exact_match, F1 (binary BCE 范式)${NC}"
echo -e "${YELLOW}  - Stick:  MAE, RMSE, direction_acc, deadzone_acc${NC}"
echo -e "${YELLOW}  - 输出: 本地 JSON (validation_ar_direct_step*.json)${NC}"
echo -e "${GREEN}========================================${NC}\n"

cleanup_gpu() {
    local exit_code=$?
    if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo -e "\n${YELLOW}检测到异常退出 (exit=$exit_code)，清理 Python 进程 $PYTHON_PID ...${NC}"
        kill "$PYTHON_PID" 2>/dev/null
        wait "$PYTHON_PID" 2>/dev/null
    fi
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo -e "${GREEN}GPU 显存已释放${NC}"
}
trap cleanup_gpu EXIT INT TERM

# python3 elefant/policy_model/validation_autoregressive_action_direct.py \
# python elefant/policy_model/validation_autoregressive_action.py \
python elefant/policy_model/validation_autoregressive_action_direct.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_FILE" \
    --data_folder "$DATA_FOLDER" \
    --n_sequences "$N_SEQUENCES" \
    $([ "$FULL_CAUSAL_MASK" = true ] && echo "--full_causal_mask") &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_EXIT=$?
unset PYTHON_PID

if [ "$PYTHON_EXIT" -ne 0 ]; then
    echo -e "${RED}验证失败 (exit=$PYTHON_EXIT)${NC}"
    exit "$PYTHON_EXIT"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}自回归验证完成！${NC}"
echo -e "${GREEN}结果: checkpoint 同目录下 validation_ar_direct_step*.json${NC}"
echo -e "${GREEN}========================================${NC}"
