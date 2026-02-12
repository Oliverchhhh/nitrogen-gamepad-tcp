#!/bin/bash
# 使用本地 dataset 目录训练 600M 模型的脚本

set -e

CONFIG_FILE="config/policy_model/600M_local_dataset.yaml"
DATA_FOLDER="dataset"  # 与 config 保持一致

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Open P2P 本地数据集训练脚本 (600M)${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}[1/3] 检查数据集路径...${NC}"
if [ ! -d "$DATA_FOLDER" ]; then
    echo -e "${RED}错误: 数据集目录不存在: $DATA_FOLDER${NC}"
    exit 1
fi
PROTO_COUNT=$(find "$DATA_FOLDER" -name "*.proto" | wc -l)
if [ "$PROTO_COUNT" -eq 0 ]; then
    echo -e "${RED}错误: 在 $DATA_FOLDER 中未找到 .proto 文件${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 找到 $PROTO_COUNT 个标注文件${NC}"

echo -e "\n${YELLOW}[2/3] 检查配置文件...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 配置文件存在${NC}"

# 检查共享内存使用情况
SHM_SIZE=$(df -h /dev/shm | tail -1 | awk '{print $2}')
SHM_AVAIL=$(df -h /dev/shm | tail -1 | awk '{print $4}')
SHM_USED=$(df -h /dev/shm | tail -1 | awk '{print $3}')
echo -e "${GREEN}共享内存 (/dev/shm): 总大小 ${SHM_SIZE}, 已用 ${SHM_USED}, 可用 ${SHM_AVAIL}${NC}"

# 清理可能残留的 PyTorch 共享内存文件
TORCH_FILES=$(find /dev/shm -name "torch_*" -type f 2>/dev/null | wc -l)
if [ "$TORCH_FILES" -gt 0 ]; then
    echo -e "${YELLOW}发现 $TORCH_FILES 个 PyTorch 共享内存文件，清理超过 10 分钟的旧文件...${NC}"
    find /dev/shm -name "torch_*" -type f -mmin +10 -delete 2>/dev/null || true
    TORCH_FILES_AFTER=$(find /dev/shm -name "torch_*" -type f 2>/dev/null | wc -l)
    echo -e "${GREEN}清理完成，剩余 $TORCH_FILES_AFTER 个文件${NC}"
fi

# ⚠️ 关键：清理 /dev/shm 中的所有 PyTorch 残留文件
echo -e "${YELLOW}清理 /dev/shm 中的 PyTorch 共享内存文件...${NC}"
TORCH_FILES_BEFORE=$(find /dev/shm -name "torch_*" -type f 2>/dev/null | wc -l)
if [ "$TORCH_FILES_BEFORE" -gt 0 ]; then
    echo -e "${YELLOW}发现 $TORCH_FILES_BEFORE 个 PyTorch 共享内存文件，全部清理...${NC}"
    find /dev/shm -name "torch_*" -type f -delete 2>/dev/null || true
    echo -e "${GREEN}✓ 清理完成${NC}"
fi

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

echo -e "\n${YELLOW}[3/3] 开始训练...${NC}"
uv run elefant/policy_model/train.py \
    --config "$CONFIG_FILE" \
    --data_folder "$DATA_FOLDER"

echo -e "\n${GREEN}训练完成！${NC}"
