#!/bin/bash
# Open P2P on NitroGen — 推理服务启动脚本
# baseline: openp2p on nitrogen (gamepad action space)
# 权重: 202603292053/checkpoint-step=00030000.ckpt

set -e

# ============================================================
# 配置变量（按需修改）
# ============================================================
# CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future.yaml" #带s0+action预测
CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset.yaml" #不带s0，纯action预测
# CHECKPOINT_PATH="202603292053/checkpoint-step=00030000.ckpt"
CHECKPOINT_PATH="202604071604_openp2p-gamepad-cuphead_all/checkpoint-step=00300000.ckpt" #纯action预测
# CHECKPOINT_PATH="202604092016_openp2p-gamepad_cuphead_all_current_GT_vjepa/checkpoint-step=00260000.ckpt" #带s0+action预测
UDS_PATH="/tmp/uds.recap"

# 推理选项
USE_FULL_INFERENCE=false   # true = 完整推理（更慢但更准）
NO_COMPILE=false           # true = 禁用 torch.compile（调试用）
USE_MANUAL_SAMPLING=false  # true = 手动采样模式
INPUT_TEXT=""              # 可选：文本提示（如游戏指令）

# ============================================================
# 颜色输出
# ============================================================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# 参数解析
# ============================================================
usage() {
    echo "用法: bash scripts/inference.sh [选项]"
    echo
    echo "选项:"
    echo "  -c    配置文件路径 (默认: $CONFIG_FILE)"
    echo "  -k    checkpoint 路径 (默认: $CHECKPOINT_PATH)"
    echo "  -u    UDS socket 路径 (默认: $UDS_PATH)"
    echo "  -t    输入文本提示 (可选)"
    echo "  --full          启用完整推理模式"
    echo "  --no-compile    禁用 torch.compile"
    echo "  --manual        启用手动采样"
    echo "  -h    显示帮助"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c) CONFIG_FILE="$2"; shift 2 ;;
        -k) CHECKPOINT_PATH="$2"; shift 2 ;;
        -u) UDS_PATH="$2"; shift 2 ;;
        -t) INPUT_TEXT="$2"; shift 2 ;;
        --full) USE_FULL_INFERENCE=true; shift ;;
        --no-compile) NO_COMPILE=true; shift ;;
        --manual) USE_MANUAL_SAMPLING=true; shift ;;
        -h|--help) usage ;;
        *) echo -e "${RED}未知参数: $1${NC}"; usage ;;
    esac
done

# ============================================================
# 预检查
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Open P2P on NitroGen — 推理服务${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}[1/4] 检查配置文件...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 配置文件: $CONFIG_FILE${NC}"

echo -e "\n${YELLOW}[2/4] 检查 checkpoint...${NC}"
if [ ! -e "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: checkpoint 不存在: $CHECKPOINT_PATH${NC}"
    echo -e "${YELLOW}提示: 请确认路径是否正确，当前工作目录: $(pwd)${NC}"
    exit 1
fi
CKPT_SIZE=$(du -sh "$CHECKPOINT_PATH" 2>/dev/null | cut -f1)
echo -e "${GREEN}✓ checkpoint: $CHECKPOINT_PATH ($CKPT_SIZE)${NC}"

echo -e "\n${YELLOW}[3/4] 检查 UDS 路径...${NC}"
if [ -e "$UDS_PATH" ]; then
    echo -e "${YELLOW}⚠ 旧 UDS 文件存在，清理: $UDS_PATH${NC}"
    rm -f "$UDS_PATH"
fi
echo -e "${GREEN}✓ UDS 路径就绪: $UDS_PATH${NC}"

echo -e "\n${YELLOW}[4/4] 检查 GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo -e "${GREEN}✓ GPU: $GPU_INFO${NC}"
else
    echo -e "${YELLOW}⚠ nvidia-smi 不可用，将尝试继续${NC}"
fi

# ============================================================
# 构建命令
# ============================================================
echo -e "\n${CYAN}========================================${NC}"
echo -e "${CYAN}启动推理服务...${NC}"
echo -e "${CYAN}========================================${NC}"

CMD=(
    python -m elefant.policy_model.inference
    --config "$CONFIG_FILE"
    --checkpoint_path "$CHECKPOINT_PATH"
)

if [ "$USE_FULL_INFERENCE" = true ]; then
    CMD+=(--use_full_inference)
    echo -e "${CYAN}  模式: 完整推理${NC}"
else
    echo -e "${CYAN}  模式: 标准推理${NC}"
fi

if [ "$NO_COMPILE" = true ]; then
    CMD+=(--no-compile)
    echo -e "${CYAN}  torch.compile: 禁用${NC}"
else
    echo -e "${CYAN}  torch.compile: 启用${NC}"
fi

if [ "$USE_MANUAL_SAMPLING" = true ]; then
    CMD+=(--use_manual_sampling)
    echo -e "${CYAN}  采样: 手动${NC}"
fi

if [ -n "$INPUT_TEXT" ]; then
    CMD+=(--input_text "$INPUT_TEXT")
    echo -e "${CYAN}  文本提示: $INPUT_TEXT${NC}"
fi

echo -e "${CYAN}  action_mapping: gamepad${NC}"
echo -e "${CYAN}  UDS: $UDS_PATH${NC}"
echo -e ""
echo -e "${CYAN}完整命令:${NC}"
echo "  ${CMD[*]}"
echo -e ""

# ============================================================
# 启动
# ============================================================
"${CMD[@]}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}推理服务已退出${NC}"
echo -e "${GREEN}========================================${NC}"
