#!/bin/bash
# 批量将 video.mp4 (1920x1080) 缩放为 video_384.mp4 (384x384)
# 使用 N 个并行进程加速，默认 16 进程
#
# 用法:
#   bash scripts/resize_videos.sh [DATA_FOLDER] [NUM_JOBS]
#
# 示例:
#   bash scripts/resize_videos.sh cuphead_dataset_converted 16

set -e

DATA_FOLDER="${1:-cuphead_dataset_converted}"
NUM_JOBS="${2:-16}"
TARGET_SIZE="384:384"
OUTPUT_NAME="video_384.mp4"

cd "$(dirname "$0")/.."

echo "========================================="
echo "批量缩放视频: $DATA_FOLDER"
echo "目标分辨率: ${TARGET_SIZE}"
echo "并行进程数: ${NUM_JOBS}"
echo "========================================="

# 收集所有需要处理的 video.mp4
mapfile -t ALL_VIDEOS < <(find "$DATA_FOLDER" -name "video.mp4" | sort)
TOTAL=${#ALL_VIDEOS[@]}
echo "找到 ${TOTAL} 个 video.mp4"

# 过滤已完成的
TODO_VIDEOS=()
for v in "${ALL_VIDEOS[@]}"; do
    out="$(dirname "$v")/${OUTPUT_NAME}"
    if [ ! -f "$out" ]; then
        TODO_VIDEOS+=("$v")
    fi
done
SKIP=$((TOTAL - ${#TODO_VIDEOS[@]}))
echo "跳过已完成: ${SKIP}，剩余: ${#TODO_VIDEOS[@]}"

if [ ${#TODO_VIDEOS[@]} -eq 0 ]; then
    echo "全部已完成！"
    exit 0
fi

# 单个视频处理函数（供 xargs 调用）
process_one() {
    local video_path="$1"
    local out_path="$(dirname "$video_path")/video_384.mp4"
    # scale=384:384 强制拉伸（squeeze resize，保留全画面）
    # -preset ultrafast -crf 18 快速编码，质量足够
    ffmpeg -y -i "$video_path" \
        -vf "scale=384:384" \
        -c:v libx264 -preset ultrafast -crf 18 \
        -an \
        "$out_path" \
        -loglevel error
    echo "OK: $out_path"
}
export -f process_one

# 用 xargs 并行执行
START_TIME=$(date +%s)
printf '%s\n' "${TODO_VIDEOS[@]}" | \
    xargs -P "$NUM_JOBS" -I{} bash -c 'process_one "$@"' _ {}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
DONE=$(find "$DATA_FOLDER" -name "$OUTPUT_NAME" | wc -l)

echo ""
echo "========================================="
echo "完成！共生成 ${DONE} 个 ${OUTPUT_NAME}"
echo "耗时: $((ELAPSED/60))m $((ELAPSED%60))s"
echo "========================================="
