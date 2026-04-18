#!/bin/bash
# 多帧动作预测训练脚本（支持单卡/多卡）
# a_in^0 预测当前帧，a_in^1 预测 t+1，a_in^2 预测 t+2，复用同一个 ActionDecoder

set -e

CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_future_action.yaml"
DATA_FOLDER="cuphead_one_level"
OUTPUT_DIR=""
TEMP_CONFIG_FILE=""
GPUS="0,1,2,3"   # 留空则使用所有可见 GPU（多卡），设为单个数字则单卡

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    echo "用法: bash scripts/train_local_dataset_future_action.sh [-d 数据集路径] [-o 输出目录] [-c 配置文件] [-g GPU列表]"
    echo
    echo "参数:"
    echo "  -d    数据集路径 (默认: cuphead_one_level)"
    echo "  -o    输出目录（覆盖配置文件中的 shared.output_path）"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_future_action.yaml)"
    echo "  -g    GPU 列表，逗号分隔 (默认: 使用所有 GPU)"
    echo "        单卡示例: -g 0"
    echo "        多卡示例: -g 0,1,2,3"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/train_local_dataset_future_action.sh              # 所有 GPU"
    echo "  bash scripts/train_local_dataset_future_action.sh -g 0         # 单卡 GPU 0"
    echo "  bash scripts/train_local_dataset_future_action.sh -g 0,1       # 双卡"
    echo "  bash scripts/train_local_dataset_future_action.sh -g 0,1,2,3   # 四卡"
}

while getopts ":d:o:c:g:h" opt; do
    case "$opt" in
        d) DATA_FOLDER="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        g) GPUS="$OPTARG" ;;
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
echo -e "${GREEN}多帧动作预测训练脚本 (n_future_action_tokens=3)${NC}"
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
echo -e "${GREEN}✓ n_future_action_tokens: ${FUTURE_TOKENS}${NC}"

# 3. 处理输出目录覆盖
TRAIN_CONFIG="$CONFIG_FILE"
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "\n${YELLOW}[3/4] 覆盖输出目录: $OUTPUT_DIR${NC}"
    mkdir -p "$OUTPUT_DIR"
    TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_train_config.XXXXXX.yaml)
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
echo -e "${GREEN}未来帧数:  ${FUTURE_TOKENS} 帧 (a_in^0..a_in^$((FUTURE_TOKENS-1)))${NC}"
echo -e "${GREEN}GPU:       ${GPU_DESC}${NC}"
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}输出目录:  $OUTPUT_DIR${NC}"
else
    echo -e "${GREEN}输出目录:  使用配置文件中的 shared.output_path${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

if [ -n "$GPUS" ]; then
    CUDA_VISIBLE_DEVICES=$GPUS python elefant/policy_model/train.py \
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

while getopts ":d:o:c:g:h" opt; do
    case "$opt" in
        d) DATA_FOLDER="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        g) GPU_ID="$OPTARG" ;;
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
echo -e "${GREEN}多帧动作预测训练脚本 (n_future_action_tokens=3)${NC}"
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
echo -e "${GREEN}✓ n_future_action_tokens: ${FUTURE_TOKENS}${NC}"

# 3. 处理输出目录覆盖
TRAIN_CONFIG="$CONFIG_FILE"
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "\n${YELLOW}[3/4] 覆盖输出目录: $OUTPUT_DIR${NC}"
    mkdir -p "$OUTPUT_DIR"
    TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_train_config.XXXXXX.yaml)
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
    TRAIN_CONFIG="$TEMP_CONFIG_FILE"
    echo -e "${GREEN}✓ 临时配置文件已生成${NC}"
else
    echo -e "\n${YELLOW}[3/4] 使用配置文件中的输出目录${NC}"
fi

# 4. 打印训练信息并启动（单卡）
echo -e "\n${YELLOW}[4/4] 启动单卡训练 (GPU ${GPU_ID})...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配置文件:  $TRAIN_CONFIG${NC}"
echo -e "${GREEN}数据集:    $DATA_FOLDER${NC}"
echo -e "${GREEN}未来帧数:  ${FUTURE_TOKENS} 帧 (a_in^0..a_in^$((FUTURE_TOKENS-1)))${NC}"
echo -e "${GREEN}GPU:       ${GPU_ID} (单卡)${NC}"
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}输出目录:  $OUTPUT_DIR${NC}"
else
    echo -e "${GREEN}输出目录:  使用配置文件中的 shared.output_path${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

CUDA_VISIBLE_DEVICES=$GPU_ID python elefant/policy_model/train.py \
    --config "$TRAIN_CONFIG" \
    --data_folder "$DATA_FOLDER"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"
