#!/bin/bash
# 对 150M_nitrogen_cuphead_all_current_gt_vjepa2 指定 step 范围跑 teacher-forcing 验证
# 范围: step=150000 ~ step=260000
# 已有 validation_step*.json 的 checkpoint 自动跳过

set -e

CKPT_DIR="output/policy_model/150M_nitrogen_cuphead_all_current_gt_vjepa2/stage3_future_vision"
CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_current.yaml"
DATA_FOLDER="NitroGen_cuphead_toy"
STEP_MIN=270000
STEP_MAX=380000

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================"
echo -e "150M_nitrogen_cuphead_all_current_gt_vjepa2 Teacher-Forcing 批量验证"
echo -e "step 范围: ${STEP_MIN} ~ ${STEP_MAX}"
echo -e "========================================${NC}"
echo "checkpoint 目录: $CKPT_DIR"
echo "数据集: $DATA_FOLDER"
echo ""

# 收集指定范围内的 checkpoint，按 step 排序
mapfile -t CKPTS < <(
    find "$CKPT_DIR" -name "checkpoint-step=*.ckpt" | sort | while read ckpt; do
        step=$(basename "$ckpt" | grep -oP '(?<=step=)\d+' | sed 's/^0*//')
        step=${step:-0}
        if [ "$step" -ge "$STEP_MIN" ] && [ "$step" -le "$STEP_MAX" ]; then
            echo "$ckpt"
        fi
    done
)

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo -e "${RED}错误: 未找到 step=${STEP_MIN}~${STEP_MAX} 范围内的 checkpoint${NC}"
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
    step=$(basename "$ckpt" | grep -oP '(?<=step=)\d+')
    val_json="$CKPT_DIR/validation_step${step}.json"

    if [ -f "$val_json" ]; then
        echo -e "${YELLOW}[跳过] step=${step} — 已有结果: $(basename $val_json)${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -e "${GREEN}[${DONE}/$((TOTAL - SKIPPED))] 验证 step=${step}...${NC}"
    START_T=$(date +%s)

    bash scripts/validate_local_dataset_current.sh \
        -k "$ckpt" \
        -d "$DATA_FOLDER" \
        -c "$CONFIG_FILE"
    EXIT_CODE=$?

    END_T=$(date +%s)
    ELAPSED=$((END_T - START_T))

    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}[失败] step=${step} (exit=$EXIT_CODE, ${ELAPSED}s)${NC}"
        FAILED=$((FAILED + 1))
    else
        echo -e "${GREEN}[完成] step=${step} (${ELAPSED}s)${NC}"
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
