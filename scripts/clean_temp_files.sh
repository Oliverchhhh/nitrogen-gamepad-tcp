#!/bin/bash
# 清理训练产生的临时文件，释放C盘空间

echo "=========================================="
echo "清理训练临时文件脚本"
echo "=========================================="

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="${PROJECT_ROOT}/.elefant_temp"

echo ""
echo "[1/3] 清理项目目录下的临时文件..."
if [ -d "$TEMP_DIR" ]; then
    echo "找到临时目录: $TEMP_DIR"
    du -sh "$TEMP_DIR" 2>/dev/null || echo "无法计算大小"
    
    read -p "是否删除临时文件? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${TEMP_DIR}/wandb"
        rm -rf "${TEMP_DIR}/torch_compiler"
        rm -rf "${TEMP_DIR}/zmq"
        echo "✓ 已清理项目临时文件"
    else
        echo "跳过清理"
    fi
else
    echo "✓ 临时目录不存在，无需清理"
fi

echo ""
echo "[2/3] 清理WSL /tmp目录中的旧文件..."
# 清理旧的/tmp目录（如果存在）
if [ -d "/tmp/elefant_wandb" ]; then
    echo "找到旧的WandB目录: /tmp/elefant_wandb"
    du -sh /tmp/elefant_wandb 2>/dev/null || echo "无法计算大小"
    
    read -p "是否删除旧的/tmp/elefant_wandb? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf /tmp/elefant_wandb
        echo "✓ 已清理 /tmp/elefant_wandb"
    fi
fi

if [ -d "/tmp/torch_compiler" ]; then
    echo "找到旧的PyTorch缓存: /tmp/torch_compiler"
    du -sh /tmp/torch_compiler 2>/dev/null || echo "无法计算大小"
    
    read -p "是否删除旧的/tmp/torch_compiler? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf /tmp/torch_compiler
        echo "✓ 已清理 /tmp/torch_compiler"
    fi
fi

if [ -d "/tmp/elefant_zmq" ]; then
    echo "找到旧的ZMQ目录: /tmp/elefant_zmq"
    du -sh /tmp/elefant_zmq 2>/dev/null || echo "无法计算大小"
    
    read -p "是否删除旧的/tmp/elefant_zmq? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf /tmp/elefant_zmq
        echo "✓ 已清理 /tmp/elefant_zmq"
    fi
fi

echo ""
echo "[3/3] 检查C盘空间..."
# 在Windows中检查C盘空间（通过WSL）
if command -v df >/dev/null 2>&1; then
    echo "当前磁盘使用情况:"
    df -h /mnt/c 2>/dev/null || echo "无法检查C盘空间"
fi

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo ""
echo "提示："
echo "1. 临时文件现在会保存在: ${TEMP_DIR}"
echo "2. 如果需要完全清理，可以删除整个 .elefant_temp 目录"
echo "3. 训练过程中，WandB数据会定期同步到云端，本地文件可以安全删除"
