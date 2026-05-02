#!/bin/bash
# Direct Action 多帧预测训练脚本（支持单卡/多卡）
# gamepad_direct 单token多帧模式：
#   - 每帧只有 1 个 a_in token（n_future_action_tokens=1）
#   - 该 token 经 MLP → 2 head 直接解码成未来 F 帧的 action
#   - button_head: BCE（12 个按钮，多标签）× F 帧
#   - stick_head:  CE（4 轴 × 3 bins）× F 帧
#   - 无扳机 head（Cuphead 数据集扳机使用率 <0.1%，视为噪声）
#   - zero_action_input: 训练和推理均不输入真实动作，消除 teacher-forcing 分布差异

set -e

CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml"
# DATA_FOLDER="cuphead_dataset_converted/v350335326"
DATA_FOLDER="cuphead_one_level_1"
OUTPUT_DIR="output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_zero_action_test"
RESUME_CKPT=""   # 指定起始 checkpoint 路径（覆盖 auto_resume）
TEMP_CONFIG_FILE=""
GPUS="3"   # 留空则使用所有可见 GPU（多卡），设为单个数字则单卡
WANDB_EXP_NAME="150M-nitrogen-cuphead-future-action-direct-F18-2head-zero-action-test"  # 覆盖 yaml 中的 wandb.exp_name（留空则使用 yaml 默认值）

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

export TORCH_COMPILE_DISABLE=1

usage() {
    echo "用法: bash scripts/train_local_dataset_future_action_direct.sh [-d 数据集路径] [-o 输出目录] [-c 配置文件] [-g GPU列表] [-k checkpoint路径]"
    echo
    echo "参数:"
    echo "  -d    数据集路径 (默认: cuphead_one_level)"
    echo "  -o    输出目录（覆盖配置文件中的 shared.output_path）"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action_direct.yaml)"
    echo "  -g    GPU 列表，逗号分隔 (默认: 使用所有 GPU)"
    echo "        单卡示例: -g 0"
    echo "        多卡示例: -g 0,1,2,3"
    echo "  -k    起始 checkpoint 路径（覆盖 auto_resume，从指定 ckpt 继续训练）"
    echo "  -w    wandb exp_name（覆盖 yaml 中的 wandb.exp_name，留空则使用 yaml 默认值）"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/train_local_dataset_future_action_direct.sh              # 所有 GPU"
    echo "  bash scripts/train_local_dataset_future_action_direct.sh -g 0         # 单卡 GPU 0"
    echo "  bash scripts/train_local_dataset_future_action_direct.sh -g 0,1       # 双卡"
    echo "  bash scripts/train_local_dataset_future_action_direct.sh -g 0,1,2,3   # 四卡"
    echo "  bash scripts/train_local_dataset_future_action_direct.sh \\"
    echo "    -o output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head_all \\"
    echo "    -k output/policy_model/150M_nitrogen_cuphead_future_action_direct_F18_2head/stage3_finetune/checkpoint-step=00030000.ckpt"
}

while getopts ":d:o:c:g:k:w:h" opt; do
    case "$opt" in
        d) DATA_FOLDER="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        g) GPUS="$OPTARG" ;;
        k) RESUME_CKPT="$OPTARG" ;;
        w) WANDB_EXP_NAME="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo -e "${RED}错误: 未知参数 -$OPTARG${NC}"; usage; exit 1 ;;
        :)  echo -e "${RED}错误: 参数 -$OPTARG 缺少值${NC}"; usage; exit 1 ;;
    esac
done

# 计算 GPU 数量
if [ -n "$GPUS" ]; then
    N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
    GPU_DESC="GPU ${GPUS} (${N_GPUS} 卡)"
else
    N_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "?")
    GPU_DESC="所有 GPU (${N_GPUS} 卡)"
fi

cleanup() {
    if [ -n "$TEMP_CONFIG_FILE" ] && [ -f "$TEMP_CONFIG_FILE" ]; then
        rm -f "$TEMP_CONFIG_FILE"
    fi
}
trap cleanup EXIT

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Direct Action 多帧预测训练脚本${NC}"
echo -e "${GREEN}action_mapping_type: gamepad_direct${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查数据集路径
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

# 2. 检查配置文件
echo -e "\n${YELLOW}[2/4] 检查配置文件...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 配置文件: $CONFIG_FILE${NC}"
FUTURE_TOKENS=$(grep "n_future_action_tokens" "$CONFIG_FILE" | awk '{print $2}')
FUTURE_FRAMES=$(grep "n_future_frames" "$CONFIG_FILE" | awk '{print $2}')
ACTION_TYPE=$(grep "action_mapping_type" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
echo -e "${GREEN}✓ action_mapping_type:    ${ACTION_TYPE}${NC}"
echo -e "${GREEN}✓ n_future_action_tokens: ${FUTURE_TOKENS} (transformer 序列中的 a_in token 数)${NC}"
echo -e "${GREEN}✓ n_future_frames:        ${FUTURE_FRAMES} (MLP head 解码的未来帧数)${NC}"

# 3. 处理输出目录覆盖 / checkpoint 注入
TRAIN_CONFIG="$CONFIG_FILE"
if [ -n "$OUTPUT_DIR" ] || [ -n "$RESUME_CKPT" ] || [ -n "$WANDB_EXP_NAME" ]; then
    echo -e "\n${YELLOW}[3/4] 生成临时配置文件...${NC}"
    TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_train_config.XXXXXX.yaml)
    cp "$CONFIG_FILE" "$TEMP_CONFIG_FILE"

    # 覆盖 output_path
    if [ -n "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        YAML_SAFE_OUTPUT_DIR=${OUTPUT_DIR//\'/\'\'}
        if ! awk -v replacement="  output_path: '$YAML_SAFE_OUTPUT_DIR'" '
            BEGIN { updated = 0 }
            {
                if (updated == 0 && $0 ~ /^[[:space:]]*output_path:[[:space:]]*/) {
                    print replacement
                    updated = 1
                    next
                }
                print
            }
            END { if (updated == 0) { exit 10 } }
        ' "$CONFIG_FILE" > "$TEMP_CONFIG_FILE"; then
            echo -e "${RED}错误: 无法在配置文件中找到 output_path 字段${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 输出目录覆盖: $OUTPUT_DIR${NC}"
    fi

    # 注入 stage3_model_path（覆盖 auto_resume）
    if [ -n "$RESUME_CKPT" ]; then
        if [ ! -f "$RESUME_CKPT" ]; then
            echo -e "${RED}错误: checkpoint 不存在: $RESUME_CKPT${NC}"
            exit 1
        fi
        YAML_SAFE_CKPT=${RESUME_CKPT//\'/\'\'}
        awk -v ckpt="    stage3_model_path: '$YAML_SAFE_CKPT'" '
            {
                if ($0 ~ /^[[:space:]]*stage3_model_path:[[:space:]]*/) {
                    print ckpt
                    next
                }
                print
            }
        ' "$TEMP_CONFIG_FILE" > "${TEMP_CONFIG_FILE}.tmp" && mv "${TEMP_CONFIG_FILE}.tmp" "$TEMP_CONFIG_FILE"
        echo -e "${GREEN}✓ 起始 checkpoint: $RESUME_CKPT${NC}"
    fi

    # 覆盖 wandb.exp_name
    if [ -n "$WANDB_EXP_NAME" ]; then
        YAML_SAFE_EXP_NAME=${WANDB_EXP_NAME//\'/\'\'}
        awk -v name="  exp_name: \"$YAML_SAFE_EXP_NAME\"" '
            {
                if ($0 ~ /^[[:space:]]*exp_name:[[:space:]]*/) {
                    print name
                    next
                }
                print
            }
        ' "$TEMP_CONFIG_FILE" > "${TEMP_CONFIG_FILE}.tmp" && mv "${TEMP_CONFIG_FILE}.tmp" "$TEMP_CONFIG_FILE"
        echo -e "${GREEN}✓ wandb exp_name: $WANDB_EXP_NAME${NC}"
    fi

    TRAIN_CONFIG="$TEMP_CONFIG_FILE"
    echo -e "${GREEN}✓ 临时配置文件已生成${NC}"
else
    echo -e "\n${YELLOW}[3/4] 使用配置文件中的输出目录${NC}"
fi

# 4. 打印训练信息并启动
echo -e "\n${YELLOW}[4/4] 启动训练 (${GPU_DESC})...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配置文件:  $TRAIN_CONFIG${NC}"
echo -e "${GREEN}数据集:    $DATA_FOLDER${NC}"
echo -e "${GREEN}动作模式:  ${ACTION_TYPE} (1 a_in token → MLP → ${FUTURE_FRAMES} 帧: button BCE + stick CE，无扳机，无 ActionDecoder)${NC}"
echo -e "${GREEN}未来帧数:  ${FUTURE_FRAMES} 帧 (1 token 解码 F 帧，n_future_action_tokens=${FUTURE_TOKENS})${NC}"
echo -e "${GREEN}GPU:       ${GPU_DESC}${NC}"
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}输出目录:  $OUTPUT_DIR${NC}"
else
    echo -e "${GREEN}输出目录:  使用配置文件中的 shared.output_path${NC}"
fi
if [ -n "$RESUME_CKPT" ]; then
    echo -e "${GREEN}起始 ckpt: $RESUME_CKPT${NC}"
else
    echo -e "${GREEN}起始 ckpt: auto_resume（自动查找最新）${NC}"
fi
if [ -n "$WANDB_EXP_NAME" ]; then
    echo -e "${GREEN}wandb:     exp_name=${WANDB_EXP_NAME}${NC}"
else
    echo -e "${GREEN}wandb:     使用 yaml 默认 exp_name${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

if [ -n "$GPUS" ]; then
    CUDA_VISIBLE_DEVICES=$GPUS \
    ELEFANT_TEMP_DIR=".elefant_temp_gpu${GPUS//,/_}" \
    python elefant/policy_model/train.py \
        --config "$TRAIN_CONFIG" \
        --data_folder "$DATA_FOLDER"
else
    python elefant/policy_model/train.py \
        --config "$TRAIN_CONFIG" \
        --data_folder "$DATA_FOLDER"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"
