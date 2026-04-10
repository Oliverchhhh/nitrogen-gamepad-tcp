#!/bin/bash
# 4 卡并行预计算 V-JEPA2.1 视觉表征
# 读取 video_384.mp4（已由 resize_videos.sh 预先缩放至 384x384）
# 用法: bash scripts/precompute_vjepa2_features.sh

set -e
cd /data2T/rjt/nitrogen-openp2p2-future-frame

CHECKPOINT="Vjepa2-1_ViT_B_16/vjepa2_1_vitb_dist_vitG_384.pt"
DATA_FOLDER="cuphead_dataset_converted"
PYTHON=".venv/bin/python"
VIDEO_NAME="video_384.mp4"   # 使用预先缩放的小分辨率视频

echo "========================================="
echo "预计算 V-JEPA2.1 表征（4 卡并行）"
echo "数据目录: $DATA_FOLDER"
echo "视频文件: $VIDEO_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "========================================="

# 检查是否已完成 resize
RESIZED=$(find "$DATA_FOLDER" -name "$VIDEO_NAME" | wc -l)
TOTAL=$(find "$DATA_FOLDER" -name "video.mp4" | wc -l)
echo "已缩放视频: ${RESIZED}/${TOTAL}"
if [ "$RESIZED" -eq 0 ]; then
    echo "ERROR: 未找到 ${VIDEO_NAME}，请先运行:"
    echo "  bash scripts/resize_videos.sh $DATA_FOLDER 16"
    exit 1
fi

# 清理旧的日志
rm -f precompute_gpu{0,1,2,3}.log

# 启动 4 个并行进程
for i in 0 1 2 3; do
    nohup $PYTHON scripts/precompute_vjepa2_features.py \
        --data_folder "$DATA_FOLDER" \
        --checkpoint "$CHECKPOINT" \
        --video_name "$VIDEO_NAME" \
        --gpu $i \
        --worker_id $i \
        --num_workers 4 \
        --batch_frames 128 \
        --skip_existing \
        > precompute_gpu$i.log 2>&1 &
    echo "GPU $i 启动，PID: $!"
done

echo ""
echo "所有进程已启动，查看进度:"
echo "  tail -f precompute_gpu0.log"
echo "  watch -n10 \"find $DATA_FOLDER -name 'vjepa2_features.pt' | wc -l\""
