#!/bin/bash
# 自回归验证脚本（nitrogen_visionencoder direct action）
# 对应评估代码: elefant/policy_model/validation_autoregressive_action_direct.py

set -e

LOG_DIR="val_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/validate_ar_action_direct_nitrogen_visionencoder_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action_direct_nitrogen_visionencoder.yaml"
# DATA_FOLDER="cuphead_one_level_4"
# DATA_FOLDER="cuphead_one_level_2"
DATA_FOLDER="NitroGen_cuphead_toy"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_nitrogen_visionencoder_F18_zero_action"
CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_nitrogen_visionencoder_F18_zero_action_v350335326/stage3_finetune/checkpoint-step=00300000.ckpt"
# CHECKPOINT_PATH="output/policy_model/150M_nitrogen_cuphead_future_action_direct_nitrogen_visionencoder_F18_zero_action_all/stage3_finetune/checkpoint-step=00300000.ckpt"
N_SEQUENCES=900
GPU_ID="1"
FULL_CAUSAL_MASK=false
MIN_STEPS=""
MAX_STEPS=""
TORCH_COMPILE_DISABLE=1

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/validate_local_dataset_autoregressive_action_direct_nitrogen_visionencoder.sh [选项]"
    echo
    echo "参数:"
    echo "  -k    checkpoint 路径（支持 .ckpt 文件、stage3_finetune 目录、或实验根目录）"
    echo "  -d    验证数据集路径 (默认: cuphead_one_level_1)"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action_direct_nitrogen_visionencoder.yaml)"
    echo "  -n    验证序列数量 (默认: 225)"
    echo "  -g    GPU ID (默认: 0)"
    echo "  -E    启用 torch.compile 模式"
    echo "  -C    禁用 torch.compile，退回 eager 模式（默认）"
    echo "  -M    使用完整 causal mask（禁用 block mask）"
    echo "  --min_steps N  最小 step（仅目录模式生效）"
    echo "  --max_steps N  最大 step（仅目录模式生效）"
    echo "  -h    显示帮助"
}

extract_step() {
    local ckpt_name
    ckpt_name="$(basename "$1")"
    if [[ "$ckpt_name" =~ step=([0-9]+)\.ckpt$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
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
        -k) CHECKPOINT_PATH="$2"; shift 2 ;;
        -d) DATA_FOLDER="$2"; shift 2 ;;
        -c) CONFIG_FILE="$2"; shift 2 ;;
        -n) N_SEQUENCES="$2"; shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        -E) TORCH_COMPILE_DISABLE=0; shift ;;
        -C) TORCH_COMPILE_DISABLE=1; shift ;;
        -M) FULL_CAUSAL_MASK=true; shift ;;
        --min_steps) MIN_STEPS="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo -e "${RED}错误: 未知参数 $1${NC}"; usage; exit 1 ;;
    esac
done

if [ -n "$MIN_STEPS" ] && [ -n "$MAX_STEPS" ] && [ "$MIN_STEPS" -gt "$MAX_STEPS" ]; then
    echo -e "${RED}错误: --min_steps 不能大于 --max_steps${NC}"
    exit 1
fi

if ! RESOLVED_CHECKPOINT_PATH=$(resolve_checkpoint_path "$CHECKPOINT_PATH"); then
    echo -e "${RED}错误: checkpoint 路径无效或未找到 .ckpt: $CHECKPOINT_PATH${NC}"
    echo -e "${YELLOW}提示: 可传入 .ckpt 文件、stage3_finetune 目录、或实验根目录${NC}"
    exit 1
fi
CHECKPOINT_PATH="$RESOLVED_CHECKPOINT_PATH"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Nitrogen VisionEncoder AR Direct Action 验证脚本${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}[1/5] 检查输入...${NC}"
[ ! -d "$DATA_FOLDER" ] && echo -e "${RED}数据集不存在: $DATA_FOLDER${NC}" && exit 1
[ ! -f "$CONFIG_FILE" ] && echo -e "${RED}配置文件不存在: $CONFIG_FILE${NC}" && exit 1
PROTO_COUNT=$(find -L "$DATA_FOLDER" -name "*.proto" | wc -l)
[ "$PROTO_COUNT" -eq 0 ] && echo -e "${RED}未找到 .proto 标注: $DATA_FOLDER${NC}" && exit 1
echo -e "${GREEN}✓ 数据集: $DATA_FOLDER ($PROTO_COUNT 个标注)${NC}"
echo -e "${GREEN}✓ 配置: $CONFIG_FILE${NC}"

echo -e "\n${YELLOW}[2/5] 校验 visionencoder 冻结配置...${NC}"
if ! python3 - <<PY
import sys
from pathlib import Path
import yaml

cfg_path = Path("$CONFIG_FILE")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

tokenizer = (((cfg or {}).get("shared") or {}).get("tokenizer") or {})
tokenizer_type = tokenizer.get("type")
ve_cfg = tokenizer.get("nitrogen_checkpoint_tokenizer_config") or {}

errors = []
if tokenizer_type != "nitrogen_checkpoint":
    errors.append(f"shared.tokenizer.type 必须为 nitrogen_checkpoint，当前为: {tokenizer_type}")
if ve_cfg.get("freeze_vision_encoder") is not True:
    errors.append("shared.tokenizer.nitrogen_checkpoint_tokenizer_config.freeze_vision_encoder 必须为 true")
if ve_cfg.get("freeze_vl_self_attention_model") is not True:
    errors.append("shared.tokenizer.nitrogen_checkpoint_tokenizer_config.freeze_vl_self_attention_model 必须为 true")

if errors:
    print("配置校验失败：")
    for e in errors:
        print(f"- {e}")
    sys.exit(1)

print("配置校验通过：vision_encoder 和 vl_self_attention_model 都处于冻结状态。")
PY
then
    echo -e "${RED}错误: visionencoder 冻结校验失败，请先修正配置${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 冻结校验通过${NC}"

echo -e "\n${YELLOW}[3/5] 设置环境...${NC}"
unset HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_ar_direct_visionencoder_validation/torch_compiler/inductor_cache"
export TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE"
echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"
echo -e "${GREEN}✓ TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE}${NC}"
echo -e "${GREEN}✓ FULL_CAUSAL_MASK=${FULL_CAUSAL_MASK}${NC}"

declare -a CKPTS=()
if [ -f "$CHECKPOINT_PATH" ]; then
    CKPTS+=("$CHECKPOINT_PATH")
else
    mapfile -t CKPTS < <(
        for ckpt in "$CHECKPOINT_PATH"/*.ckpt; do
            [ -e "$ckpt" ] || continue
            step="$(extract_step "$ckpt")" || continue
            step_num=$((10#$step))
            if [ -n "$MIN_STEPS" ] && [ "$step_num" -lt "$MIN_STEPS" ]; then
                continue
            fi
            if [ -n "$MAX_STEPS" ] && [ "$step_num" -gt "$MAX_STEPS" ]; then
                continue
            fi
            printf "%d\t%s\n" "$step_num" "$ckpt"
        done | sort -n | awk -F'\t' '{print $2}'
    )
fi

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo -e "${RED}错误: 未找到可验证的 checkpoint${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[4/5] checkpoint 列表...${NC}"
for ckpt in "${CKPTS[@]}"; do
    step="$(extract_step "$ckpt" || echo "unknown")"
    echo -e "${GREEN}  - step=${step}  $(basename "$ckpt")${NC}"
done

echo -e "\n${YELLOW}[5/5] 开始自回归验证...${NC}"
TOTAL=${#CKPTS[@]}
DONE=0
FAILED=0

for ckpt in "${CKPTS[@]}"; do
    step="$(extract_step "$ckpt" || echo "unknown")"
    echo -e "\n${GREEN}----------------------------------------${NC}"
    echo -e "${GREEN}[$((DONE + FAILED + 1))/$TOTAL] 验证 checkpoint step=${step}${NC}"
    echo -e "${GREEN}checkpoint: $ckpt${NC}"
    echo -e "${GREEN}----------------------------------------${NC}"

    rm -rf /tmp/elefant_zmq /ephemeral/elefant_tmp_data /tmp/elefant_data 2>/dev/null || true

    CMD=(python3 elefant/policy_model/validation_autoregressive_action_direct.py
        --checkpoint_path "$ckpt"
        --config_path "$CONFIG_FILE"
        --data_folder "$DATA_FOLDER"
        --n_sequences "$N_SEQUENCES"
    )
    [ "$FULL_CAUSAL_MASK" = true ] && CMD+=(--full_causal_mask)

    if "${CMD[@]}"; then
        DONE=$((DONE + 1))
    else
        FAILED=$((FAILED + 1))
        echo -e "${RED}step=${step} 验证失败${NC}"
    fi

    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}AR direct 验证完成${NC}"
echo -e "${GREEN}成功: $DONE / $TOTAL${NC}"
if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}失败: $FAILED${NC}"
    exit 1
fi
echo -e "${GREEN}结果: checkpoint 同目录下 validation_ar_direct_step*.json${NC}"
echo -e "${GREEN}========================================${NC}"
