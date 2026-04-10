#!/bin/bash
# 自回归验证脚本 — 逐帧用模型自己采样的 action 作为下一步输入
# 比 teacher-forcing validation 更贴近真实推理效果

set -e

# 配置变量
CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_current.yaml"
DATA_FOLDER="NitroGen_cuphead_toy"
CHECKPOINT_PATH="output_20260401/policy_model/150M_nitrogen/stage3_finetune/checkpoint-step=00040000.ckpt"
N_SEQUENCES=72
TEMPERATURE=1.0

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/validate_local_dataset_autoregressive.sh -k checkpoint路径 [-d 数据集] [-c 配置] [-n 序列数] [-t 温度]"
    echo
    echo "参数:"
    echo "  -k    checkpoint 路径（必填）"
    echo "  -d    验证数据集路径 (默认: NitroGen_cuphead_toy)"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_current.yaml)"
    echo "  -n    验证的视频序列数量 (默认: 72，对应 NitroGen_cuphead_toy 全部独立样本)"
    echo "  -t    采样温度 (默认: 1.0)"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/validate_local_dataset_autoregressive.sh \\"
    echo "    -k output/policy_model/150M_nitrogen_current/stage3_future_vision/checkpoint-step=00040000.ckpt"
    echo
    echo "  bash scripts/validate_local_dataset_autoregressive.sh \\"
    echo "    -k path/to/ckpt -d NitroGen_cuphead_toy -n 20 -t 0.8"
    echo
    echo "输出:"
    echo "  - 本地 JSON: checkpoint 同目录下 validation_ar_step{STEP}.json"
}

while getopts ":k:d:c:n:t:h" opt; do
    case "$opt" in
        k) CHECKPOINT_PATH="$OPTARG" ;;
        d) DATA_FOLDER="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        n) N_SEQUENCES="$OPTARG" ;;
        t) TEMPERATURE="$OPTARG" ;;
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
echo -e "${GREEN}Open P2P 自回归验证脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查
echo -e "\n${YELLOW}[1/3] 检查输入...${NC}"
[ ! -d "$DATA_FOLDER" ] && echo -e "${RED}数据集不存在: $DATA_FOLDER${NC}" && exit 1
[ ! -f "$CONFIG_FILE" ] && echo -e "${RED}配置文件不存在: $CONFIG_FILE${NC}" && exit 1
[ ! -f "$CHECKPOINT_PATH" ] && echo -e "${RED}checkpoint 不存在: $CHECKPOINT_PATH${NC}" && exit 1

PROTO_COUNT=$(find -L "$DATA_FOLDER" -name "*.proto" | wc -l)
echo -e "${GREEN}✓ 数据集: $DATA_FOLDER ($PROTO_COUNT 个标注)${NC}"
echo -e "${GREEN}✓ 配置: $CONFIG_FILE${NC}"
echo -e "${GREEN}✓ checkpoint: $CHECKPOINT_PATH${NC}"

# 环境
echo -e "\n${YELLOW}[2/3] 设置环境...${NC}"
unset HF_ENDPOINT
export CUDA_VISIBLE_DEVICES=1

# torch.compile 编译缓存
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_ar_validation/torch_compiler/inductor_cache"

echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"
echo -e "${GREEN}✓ compile 缓存: ${TORCHINDUCTOR_CACHE_DIR}${NC}"

# 运行
echo -e "\n${YELLOW}[3/3] 开始自回归验证...${NC}"
echo -e "${GREEN}序列数: $N_SEQUENCES, 温度: $TEMPERATURE${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 记录 PID 以便异常退出时清理 GPU 显存
cleanup_gpu() {
    local exit_code=$?
    if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo -e "\n${YELLOW}检测到异常退出 (exit=$exit_code)，清理 Python 进程 $PYTHON_PID ...${NC}"
        kill "$PYTHON_PID" 2>/dev/null
        wait "$PYTHON_PID" 2>/dev/null
    fi
    # 强制释放 GPU 显存
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo -e "${GREEN}GPU 显存已释放${NC}"
}

trap cleanup_gpu EXIT INT TERM

python3 elefant/policy_model/validation_autoregressive.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_FILE" \
    --data_folder "$DATA_FOLDER" \
    --n_sequences "$N_SEQUENCES" \
    --sampling_temperature "$TEMPERATURE" &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_EXIT=$?
unset PYTHON_PID  # 正常退出，不需要 cleanup 再 kill

if [ "$PYTHON_EXIT" -ne 0 ]; then
    echo -e "${RED}验证失败 (exit=$PYTHON_EXIT)${NC}"
    exit "$PYTHON_EXIT"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}自回归验证完成！${NC}"
echo -e "${GREEN}结果: checkpoint 同目录下 validation_ar_step*.json${NC}"
echo -e "${GREEN}========================================${NC}"
