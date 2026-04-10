#!/bin/bash
# 循环对 150M_nitrogen_cuphead_all 所有 checkpoint 跑自回归验证
# 已有 validation_ar_step*.json 的 checkpoint 自动跳过

set -e

CKPT_DIR="output_20260401/policy_model/150M_nitrogen_cuphead_all/stage3_finetune"
CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_current.yaml"
DATA_FOLDER="NitroGen_cuphead_toy"
N_SEQUENCES=72
TEMPERATURE=1.0

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================"
echo -e "150M_nitrogen_cuphead_all 自回归批量验证"
echo -e "========================================${NC}"
echo "checkpoint 目录: $CKPT_DIR"
echo "数据集: $DATA_FOLDER"
echo "序列数: $N_SEQUENCES, 温度: $TEMPERATURE"
echo ""

# 收集所有 checkpoint，按 step 排序
mapfile -t CKPTS < <(find "$CKPT_DIR" -name "checkpoint-step=*.ckpt" | sort)

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo -e "${RED}错误: 未找到任何 checkpoint${NC}"
    exit 1
fi

echo "找到 ${#CKPTS[@]} 个 checkpoint:"
for ckpt in "${CKPTS[@]}"; do
    echo "  $(basename $ckpt)"
done
echo ""

TOTAL=${#CKPTS[@]}
DONE=0
SKIPPED=0
FAILED=0

for ckpt in "${CKPTS[@]}"; do
    # 从路径提取 step 编号
    step=$(basename "$ckpt" | grep -oP '(?<=step=)\d+')
    ar_json="$CKPT_DIR/validation_ar_step${step}.json"

    # 已有结果则跳过
    if [ -f "$ar_json" ]; then
        echo -e "${YELLOW}[跳过] step=${step} — 已有结果: $(basename $ar_json)${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -e "${GREEN}[${DONE}/$((TOTAL - SKIPPED))] 验证 step=${step}...${NC}"
    START_T=$(date +%s)

    bash scripts/validate_local_dataset_autoregressive.sh \
        -k "$ckpt" \
        -d "$DATA_FOLDER" \
        -c "$CONFIG_FILE" \
        -n "$N_SEQUENCES" \
        -t "$TEMPERATURE"
    EXIT_CODE=$?

    END_T=$(date +%s)
    ELAPSED=$((END_T - START_T))

    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}[失败] step=${step} (exit=$EXIT_CODE, ${ELAPSED}s)${NC}"
        FAILED=$((FAILED + 1))
    else
        echo -e "${GREEN}[完成] step=${step} (${ELAPSED}s) → $(basename $ar_json)${NC}"
        DONE=$((DONE + 1))
    fi
    echo ""
done

echo -e "${GREEN}========================================"
echo -e "批量验证完成"
echo -e "  完成: $DONE  跳过: $SKIPPED  失败: $FAILED"
echo -e "========================================${NC}"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
