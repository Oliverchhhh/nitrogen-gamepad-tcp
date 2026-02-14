#!/bin/bash
# 使用本地 dataset 目录训练模型的脚本（V-JEPA2 版本）

set -e  # 遇到错误立即退出

# 配置变量
CONFIG_FILE="config/policy_model/150M_local_dataset_vjepa2.yaml"
DATA_FOLDER="dataset"  # 数据集路径（相对于项目根目录）
OUTPUT_DIR="./output"  # 输出目录（可选）

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Open P2P 本地数据集训练脚本 (V-JEPA2)${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查数据集路径
echo -e "\n${YELLOW}[1/5] 检查数据集路径...${NC}"
if [ ! -d "$DATA_FOLDER" ]; then
    echo -e "${RED}错误: 数据集目录不存在: $DATA_FOLDER${NC}"
    echo -e "${YELLOW}提示: 请确保数据集已下载到 $DATA_FOLDER${NC}"
    exit 1
fi

# 检查是否有 .proto 文件
PROTO_COUNT=$(find "$DATA_FOLDER" -name "*.proto" | wc -l)
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

# 3. 检查 GPU
echo -e "\n${YELLOW}[3/5] 检查 GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU 检查完成${NC}"
    echo -e "${YELLOW}⚠️  注意: V-JEPA2 模型较大，建议至少 32GB 显存${NC}"
else
    echo -e "${YELLOW}警告: 未检测到 nvidia-smi，可能没有 GPU${NC}"
fi

# 4. 设置环境变量和检查共享内存
echo -e "\n${YELLOW}[4/5] 设置环境变量和检查共享内存...${NC}"

# 如果设置了 HF 镜像，取消设置（避免影响下载）
unset HF_ENDPOINT

# torch.compile 编译缓存：使用项目内目录，第二次及以后运行可复用，避免重复编译
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=".elefant_temp/torch_compiler/inductor_cache"

# V-JEPA2 相关环境变量（如果需要）
# 如果 V-JEPA2 代码不在默认路径，可以通过环境变量指定
# export VJEPA2_PATH="/path/to/vjepa2"

# 设置 CUDA 设备（如果需要指定特定 GPU）
# export CUDA_VISIBLE_DEVICES=0

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
echo -e "\n${YELLOW}[5/5] 开始训练...${NC}"
echo -e "${GREEN}配置文件: $CONFIG_FILE${NC}"
echo -e "${GREEN}数据集路径: $DATA_FOLDER${NC}"
echo -e "${GREEN}视觉编码器: V-JEPA2${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 使用 uv 运行训练脚本
# 注意：--data_folder 参数会覆盖配置文件中的 local_prefix
uv run elefant/policy_model/train.py \
    --config "$CONFIG_FILE" \
    --data_folder "$DATA_FOLDER"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"
