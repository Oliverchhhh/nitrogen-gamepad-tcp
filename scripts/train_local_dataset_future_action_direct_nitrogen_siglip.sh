#!/bin/bash
# nitrogen_siglip tokenizer 训练脚本（dense visual tokens，不做 pooling）

set -e

CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action_direct_nitrogen_siglip.yaml"
# DATA_FOLDER="cuphead_dataset_converted/v350335326"
DATA_FOLDER="cuphead_one_level_1"
OUTPUT_DIR="output/policy_model/150M_nitrogen_cuphead_future_action_direct_siglip_F5_onelevel1"
RESUME_CKPT=""
VISION_ENCODER_NAME=""
VISION_ENCODER_LOCAL_PATH=""
WANDB_EXP_NAME="150M_nitrogen_cuphead_future_action_direct_siglip_F5_onelevel1"
GPUS="3"
TEMP_CONFIG_FILE=""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/train_local_dataset_future_action_direct_nitrogen_siglip.sh [选项]"
    echo
    echo "参数:"
    echo "  -d    数据集目录 (默认: cuphead_dataset_converted)"
    echo "  -c    配置文件 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action_direct_nitrogen_siglip.yaml)"
    echo "  -g    GPU 列表，逗号分隔 (默认: 0)"
    echo "  -o    输出目录，覆盖 shared.output_path"
    echo "  -k    从指定 checkpoint 继续训练（覆盖 stage3_model_path）"
    echo "  -v    视觉编码器名称（覆盖 tokenizer.nitrogen_siglip_tokenizer_config.vision_encoder_name）"
    echo "  -p    本地视觉编码器目录（优先于 -v，例如 ./checkpoints/siglip2-large-patch16-256）"
    echo "  -w    wandb exp_name（覆盖 yaml 中配置）"
    echo "  -h    显示帮助"
}

while getopts ":d:c:g:o:k:v:p:w:h" opt; do
    case "$opt" in
        d) DATA_FOLDER="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        g) GPUS="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        k) RESUME_CKPT="$OPTARG" ;;
        v) VISION_ENCODER_NAME="$OPTARG" ;;
        p) VISION_ENCODER_LOCAL_PATH="$OPTARG" ;;
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
echo -e "${GREEN}Nitrogen SigLIP Direct Action 训练脚本${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}[1/4] 检查输入...${NC}"
[ ! -d "$DATA_FOLDER" ] && echo -e "${RED}数据集不存在: $DATA_FOLDER${NC}" && exit 1
[ ! -f "$CONFIG_FILE" ] && echo -e "${RED}配置文件不存在: $CONFIG_FILE${NC}" && exit 1
[ -n "$RESUME_CKPT" ] && [ ! -f "$RESUME_CKPT" ] && echo -e "${RED}checkpoint 不存在: $RESUME_CKPT${NC}" && exit 1
[ -n "$VISION_ENCODER_LOCAL_PATH" ] && [ ! -d "$VISION_ENCODER_LOCAL_PATH" ] && echo -e "${RED}本地视觉编码器目录不存在: $VISION_ENCODER_LOCAL_PATH${NC}" && exit 1

PROTO_COUNT=$(find -L "$DATA_FOLDER" -name "*.proto" | wc -l)
[ "$PROTO_COUNT" -eq 0 ] && echo -e "${RED}未找到 .proto 标注: $DATA_FOLDER${NC}" && exit 1
echo -e "${GREEN}✓ 数据集: $DATA_FOLDER ($PROTO_COUNT 个标注)${NC}"
echo -e "${GREEN}✓ 配置: $CONFIG_FILE${NC}"

echo -e "\n${YELLOW}[2/4] 生成训练配置...${NC}"
TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_nitrogen_siglip.XXXXXX.yaml)

python3 - "$CONFIG_FILE" "$TEMP_CONFIG_FILE" "$OUTPUT_DIR" "$RESUME_CKPT" "$VISION_ENCODER_NAME" "$VISION_ENCODER_LOCAL_PATH" "$WANDB_EXP_NAME" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
output_dir = sys.argv[3]
resume_ckpt = sys.argv[4]
vision_encoder_name = sys.argv[5]
vision_encoder_local_path = sys.argv[6]
wandb_exp_name = sys.argv[7]

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

if output_dir:
    cfg["shared"]["output_path"] = output_dir

if resume_ckpt:
    cfg.setdefault("stage3_finetune", {}).setdefault("init", {})
    cfg["stage3_finetune"]["init"]["stage3_model_path"] = resume_ckpt

if vision_encoder_name:
    cfg["shared"]["tokenizer"]["nitrogen_siglip_tokenizer_config"]["vision_encoder_name"] = vision_encoder_name

if vision_encoder_local_path:
    cfg["shared"]["tokenizer"]["nitrogen_siglip_tokenizer_config"]["vision_encoder_local_path"] = vision_encoder_local_path

if wandb_exp_name:
    cfg.setdefault("wandb", {})["exp_name"] = wandb_exp_name

dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY

echo -e "${GREEN}✓ 临时配置: $TEMP_CONFIG_FILE${NC}"

echo -e "\n${YELLOW}[3/4] 环境设置...${NC}"
unset HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPUS"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_nitrogen_siglip/torch_compiler/inductor_cache"
echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

echo -e "\n${YELLOW}[4/4] 开始训练...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}模型: nitrogen_siglip + gamepad_direct + F5 + zero_action_input${NC}"
echo -e "${GREEN}训练配置: $TEMP_CONFIG_FILE${NC}"
echo -e "${GREEN}数据集: $DATA_FOLDER${NC}"
if [ -n "$VISION_ENCODER_LOCAL_PATH" ]; then
    echo -e "${GREEN}视觉编码器: 本地目录 $VISION_ENCODER_LOCAL_PATH${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

python3 elefant/policy_model/train.py \
    --config "$TEMP_CONFIG_FILE" \
    --data_folder "$DATA_FOLDER"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}训练完成${NC}"
echo -e "${GREEN}========================================${NC}"
