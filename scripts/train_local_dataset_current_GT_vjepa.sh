#!/bin/bash
# 使用本地 dataset 目录训练 Current Vision（输入 conv，state GT 使用 vjepa2.1）的 Stage3 模型脚本

set -e  # 遇到错误立即退出

# 配置变量
CONFIG_FILE="config/policy_model/150M_local_nitrogen_dataset_current.yaml"
DATA_FOLDER="dataset"  # 数据集路径（相对于项目根目录）
OUTPUT_DIR=""  # 输出目录（可选，不传则使用配置文件中的 shared.output_path）
TEMP_CONFIG_FILE=""
NO_COMPILE=false

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    echo "用法: bash scripts/train_local_dataset_current_GT_vjepa.sh [-d 数据集路径] [-o 输出目录] [-c 配置文件] [-n]"
    echo
    echo "参数:"
    echo "  -d    数据集路径 (默认: dataset)"
    echo "  -o    输出目录。传入后将覆盖配置文件中的 shared.output_path"
    echo "  -c    配置文件路径 (默认: config/policy_model/150M_local_nitrogen_dataset_current.yaml)"
    echo "  -n    禁用 torch.compile（向 train_future.py 透传 --no_compile）"
    echo "  -h    显示帮助"
    echo
    echo "示例:"
    echo "  bash scripts/train_local_dataset_current_GT_vjepa.sh"
    echo "  bash scripts/train_local_dataset_current_GT_vjepa.sh -c config/policy_model/150M_local_nitrogen_dataset_current.yaml -d NitroGen_cuphead"
    echo "  bash scripts/train_local_dataset_current_GT_vjepa.sh -c config/policy_model/150M_local_nitrogen_dataset_current.yaml -d NitroGen_cuphead -n"
}

while getopts ":d:o:c:nh" opt; do
    case "$opt" in
        d) DATA_FOLDER="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        n) NO_COMPILE=true ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo -e "${RED}错误: 未知参数 -$OPTARG${NC}"
            usage
            exit 1
            ;;
        :)
            echo -e "${RED}错误: 参数 -$OPTARG 缺少值${NC}"
            usage
            exit 1
            ;;
    esac
done

cleanup() {
    if [ -n "$TEMP_CONFIG_FILE" ] && [ -f "$TEMP_CONFIG_FILE" ]; then
        rm -f "$TEMP_CONFIG_FILE"
    fi
}

trap cleanup EXIT

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Open P2P Current 训练脚本（GT: VJEPA2.1）${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查数据集路径
echo -e "\n${YELLOW}[1/5] 检查数据集路径...${NC}"
if [ ! -d "$DATA_FOLDER" ]; then
    echo -e "${RED}错误: 数据集目录不存在: $DATA_FOLDER${NC}"
    echo -e "${YELLOW}提示: 请确保数据集已下载到 $DATA_FOLDER${NC}"
    exit 1
fi

# 检查是否有 .proto 文件（-L 可跟随软链接目录）
PROTO_COUNT=$(find -L "$DATA_FOLDER" -name "*.proto" | wc -l)
if [ "$PROTO_COUNT" -eq 0 ]; then
    echo -e "${RED}错误: 在 $DATA_FOLDER 中未找到 .proto 文件${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 找到 $PROTO_COUNT 个标注文件${NC}"

# 2. 检查配置文件
echo -e "\n${YELLOW}[2/5] 检查配置文件...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 配置文件存在${NC}"

# 2.5 处理输出目录覆盖（通过临时配置文件实现）
TRAIN_CONFIG="$CONFIG_FILE"
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}检测到输出目录参数，覆盖 shared.output_path: $OUTPUT_DIR${NC}"
    mkdir -p "$OUTPUT_DIR"
    TEMP_CONFIG_FILE=$(mktemp /tmp/open_p2p_current_gt_vjepa_train_config.XXXXXX.yaml)

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
        END {
            if (updated == 0) {
                exit 10
            }
        }
    ' "$CONFIG_FILE" > "$TEMP_CONFIG_FILE"; then
        echo -e "${RED}错误: 无法在配置文件中找到 output_path 字段，无法覆盖输出目录${NC}"
        exit 1
    fi

    TRAIN_CONFIG="$TEMP_CONFIG_FILE"
    echo -e "${GREEN}✓ 已生成临时配置文件: $TRAIN_CONFIG${NC}"
fi

# 3. 检查 GPU
echo -e "\n${YELLOW}[3/5] 检查 GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU 检查完成${NC}"
else
    echo -e "${YELLOW}警告: 未检测到 nvidia-smi，可能没有 GPU${NC}"
fi

# 4. 设置环境变量和检查共享内存
echo -e "\n${YELLOW}[4/5] 设置环境变量和检查共享内存...${NC}"

# 如果设置了 HF 镜像，取消设置（避免影响下载）
unset HF_ENDPOINT

# torch.compile 编译缓存：使用项目内目录，第二次及以后运行可复用
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp_current_gt_vjepa/torch_compiler/inductor_cache"

# 设置 CUDA 设备（与 train_local_dataset.sh 对齐，默认单卡避免多卡环境异常）
export CUDA_VISIBLE_DEVICES=1
echo -e "${GREEN}单卡模式: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

# ⚠️ 关键：清理 /dev/shm 中的所有 PyTorch 残留文件
echo -e "${YELLOW}清理 /dev/shm 中的 PyTorch 共享内存文件...${NC}"
TORCH_FILES_BEFORE=$(find /dev/shm -name "torch_*" -type f 2>/dev/null | wc -l)
if [ "$TORCH_FILES_BEFORE" -gt 0 ]; then
    echo -e "${YELLOW}发现 $TORCH_FILES_BEFORE 个 PyTorch 共享内存文件，全部清理...${NC}"
    find /dev/shm -name "torch_*" -type f -delete 2>/dev/null || true
    echo -e "${GREEN}✓ 清理完成${NC}"
fi

# 检查 /dev/shm 使用情况
SHM_SIZE=$(df -h /dev/shm | tail -1 | awk '{print $2}')
SHM_AVAIL=$(df -h /dev/shm | tail -1 | awk '{print $4}')
SHM_USED=$(df -h /dev/shm | tail -1 | awk '{print $3}')
echo -e "${GREEN}共享内存 (/dev/shm): 总大小 ${SHM_SIZE}, 已用 ${SHM_USED}, 可用 ${SHM_AVAIL}${NC}"

# 如果可用空间小于 20GB，警告用户
SHM_AVAIL_GB=$(df /dev/shm | tail -1 | awk '{print int($4/1024/1024)}')
if [ "$SHM_AVAIL_GB" -lt 20 ]; then
    echo -e "${RED}警告: /dev/shm 可用空间不足 20GB！${NC}"
    echo -e "${YELLOW}建议在 Windows 的 .wslconfig 中添加：kernelCommandLine = shmsize=32g${NC}"
    echo -e "${YELLOW}然后运行 'wsl --shutdown' 重启 WSL${NC}"
    read -p "是否继续训练？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ 环境变量设置完成${NC}"

# 5. 开始训练
echo -e "\n${YELLOW}[5/5] 开始训练 (Current + GT VJEPA2.1)...${NC}"
echo -e "${GREEN}配置文件: $TRAIN_CONFIG${NC}"
echo -e "${GREEN}数据集路径: $DATA_FOLDER${NC}"
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
else
    echo -e "${GREEN}输出目录: 使用配置文件中的 shared.output_path${NC}"
fi
if [ "$NO_COMPILE" = true ]; then
    echo -e "${GREEN}训练模式: no_compile${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

# 使用 train_future.py 入口，配合 current 配置进行 current target 训练
# 注意：--data_folder 参数会覆盖配置文件中的 local_prefix
CMD=(python elefant/policy_model/train_future.py --config "$TRAIN_CONFIG" --data_folder "$DATA_FOLDER")
if [ "$NO_COMPILE" = true ]; then
    CMD+=(--no_compile)
fi
"${CMD[@]}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Current + GT VJEPA2.1 训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"
