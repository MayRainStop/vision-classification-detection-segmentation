#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/flower_classification_task1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

PY=/root/miniconda3/bin/python
RESULT_ROOT=experiment_results
mkdir -p "$RESULT_ROOT/logs"

echo "===== EXTENDED CNN GRID START $(date '+%F %T') =====" | tee -a "$RESULT_ROOT/logs/extended_cnn_grid.log"
nvidia-smi | tee -a "$RESULT_ROOT/logs/extended_cnn_grid.log" || true

$PY -m flower_classification.grid_search \
  --data-root . \
  --output-root "$RESULT_ROOT/extended_cnn_grid" \
  --models resnet50,resnet101,efficientnet_b0,efficientnet_b3,convnext_tiny \
  --scratch-models= \
  --epochs 200 \
  --patience 15 \
  --num-workers 8 \
  --learning-rates 0.0003,0.0001,0.00005 \
  --weight-decays 0.0001,0.00001 \
  --batch-sizes 16,32 \
  --label-smoothing 0,0.1 \
  2>&1 | tee -a "$RESULT_ROOT/logs/extended_cnn_grid.log"

echo "===== EXTENDED CNN GRID DONE $(date '+%F %T') =====" | tee -a "$RESULT_ROOT/logs/extended_cnn_grid.log"
