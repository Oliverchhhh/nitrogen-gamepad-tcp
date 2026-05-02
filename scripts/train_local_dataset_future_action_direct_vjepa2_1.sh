#!/bin/bash
# vjepa2.1 + gamepad_direct + F18 + zero_action_input 训练脚本

set -e

CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action_direct_vjepa2_1.yaml"
DATA_FOLDER="cuphead_dataset_converted/v350335326"
OUTPUT_DIR=""
RESUME_CKPT=""
VJEPA_CKPT="Vjepa2-1_ViT_B_16/vjepa2_1_vitb_dist_vitG_384.pt"
GPUS="2"
WANDB_EXP_NAME=""
TEMP_CONFIG_FILE=""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/train_local_dataset_future_action_direct_vjepa2_1.sh [选项]"
    echo
    echo "参数:"
    echo "  -d    数据集目录 (默认: cuphead_dataset_converted)"
    echo "  -c    配置文件 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action_direct_vjepa2_1.yaml)"
    echo "  -g    GPU 列表，逗号分隔 (默认: 0)"
    echo "  -o    输出目录，覆盖 shared.output_path"
    echo "  -k    从指定 checkpoint 继续训练（覆盖 stage3_model_path）"
    echo "  -v    V-JEPA2.1 checkpoint 路径（覆盖 tokenizer.vjepa_tokenizer_config.checkpoint_path）"
    echo "  -w    wandb exp_name（覆盖 yaml 中配置）"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/train_local_dataset_future_action_direct_vjepa2_1.sh \\"
    echo "    -d cuphead_dataset_converted -g 0 \\"
    echo "    -v data/pretrain_models/ViT-B16/vjepa2_1_vitb_dist_vitG_384.pt"
}

while getopts ":d:c:g:o:k:v:w:h" opt; do
    case "$opt" in
        d) DATA_FOLDER="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        g) GPUS="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        k) RESUME_CKPT="$OPTARG" ;;
        v) VJEPA_CKPT="$OPTARG" ;;
        w) WANDB_EXP_NAME="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo -e "${RED}错误: 未知参数 -$OPTARG${NC}"; usage; exit 1 ;;
        :)  echo -e "${RED}错误: 参数 -$OPTARG 缺少值${NC}"; usage; exit 1 ;;
    esac
done

cleanup() {
    if [ -n "$TEMP_CONFIG_FILE" ] && [ -f "$TEMP_CONFIG_FILE" ]; then
        rm -f "$TEMP_CONFIG_FILE"
    fi
}
trap cleanup EXIT

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}V-JEPA2.1 Direct Action 训练脚本${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}[1/4] 检查输入...${NC}"
[ ! -d "$DATA_FOLDER" ] && echo -e "${RED}数据集不存在: $DATA_FOLDER${NC}" && exit 1
[ ! -f "$CONFIG_FILE" ] && echo -e "${RED}配置文件不存在: $CONFIG_FILE${NC}" && exit 1
[ -n "$RESUME_CKPT" ] && [ ! -f "$RESUME_CKPT" ] && echo -e "${RED}checkpoint 不存在: $RESUME_CKPT${NC}" && exit 1
[ -n "$VJEPA_CKPT" ] && [ ! -f "$VJEPA_CKPT" ] && echo -e "${RED}V-JEPA ckpt 不存在: $VJEPA_CKPT${NC}" && exit 1

PROTO_COUNT=$(find -L "$DATA_FOLDER" -name "*.proto" | wc -l)
[ "$PROTO_COUNT" -eq 0 ] && echo -e "${RED}未找到 .proto 标注: $DATA_FOLDER${NC}" && exit 1
echo -e "${GREEN}✓ 数据集: $DATA_FOLDER ($PROTO_COUNT 个标注)${NC}"
echo -e "${GREEN}✓ 配置: $CONFIG_FILE${NC}"

echo -e "\n${YELLOW}[2/4] 生成训练配置...${NC}"
TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_vjepa2_1_direct.XXXXXX.yaml)

python3 - "$CONFIG_FILE" "$TEMP_CONFIG_FILE" "$OUTPUT_DIR" "$RESUME_CKPT" "$VJEPA_CKPT" "$WANDB_EXP_NAME" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
output_dir = sys.argv[3]
resume_ckpt = sys.argv[4]
vjepa_ckpt = sys.argv[5]
wandb_exp_name = sys.argv[6]

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

if output_dir:
    cfg["shared"]["output_path"] = output_dir

if resume_ckpt:
    cfg.setdefault("stage3_finetune", {}).setdefault("init", {})
    cfg["stage3_finetune"]["init"]["stage3_model_path"] = resume_ckpt

if vjepa_ckpt:
    cfg["shared"]["tokenizer"]["vjepa_tokenizer_config"]["checkpoint_path"] = vjepa_ckpt

if wandb_exp_name:
    cfg.setdefault("wandb", {})["exp_name"] = wandb_exp_name

dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY

echo -e "${GREEN}✓ 临时配置: $TEMP_CONFIG_FILE${NC}"

echo -e "\n${YELLOW}[3/4] 环境设置...${NC}"
unset HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPUS"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_vjepa2_1_direct/torch_compiler/inductor_cache"
echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

echo -e "\n${YELLOW}[4/4] 开始训练...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}模型: vjepa2.1 + gamepad_direct + F18 + zero_action_input${NC}"
echo -e "${GREEN}训练配置: $TEMP_CONFIG_FILE${NC}"
echo -e "${GREEN}数据集: $DATA_FOLDER${NC}"
echo -e "${GREEN}========================================${NC}\n"

python3 elefant/policy_model/train.py \
    --config "$TEMP_CONFIG_FILE" \
    --data_folder "$DATA_FOLDER"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}训练完成${NC}"
echo -e "${GREEN}========================================${NC}"
