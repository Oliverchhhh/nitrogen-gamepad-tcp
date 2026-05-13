#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"   # 自动定位为本脚本所在目录
DATA_DIR="/path/to/cuphead"                  # 修改为实际数据目录

docker run --rm -it --gpus all \
  --network=host \
  -e CUDA_MPS_PIPE_DIRECTORY="" \
  -v ${REPO_DIR}/checkpoints:/app/checkpoints \
  -v ${REPO_DIR}/config:/app/config \
  -v ${REPO_DIR}/elefant:/app/elefant \
  -v ${REPO_DIR}/output:/app/output \
  -v ${DATA_DIR}:/app/cuphead_dataset_converted \
  open-p2p-gamepad \
  .venv/bin/python elefant/policy_model/train_stamo.py \
    --config config/policy_model/150M_stamo_cotrain.yaml \
    --data_folder /app/cuphead_dataset_converted
