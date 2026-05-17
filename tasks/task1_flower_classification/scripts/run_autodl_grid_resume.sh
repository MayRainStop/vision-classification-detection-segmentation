#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/flower_classification_task1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
RESULT_ROOT=experiment_results
mkdir -p "$RESULT_ROOT/logs"
echo "===== GRID RESUME START $(date '+%F %T') =====" | tee -a "$RESULT_ROOT/logs/resume.log"
nvidia-smi | tee -a "$RESULT_ROOT/logs/resume.log" || true
echo "===== CORE CNN GRID RESUME $(date '+%F %T') =====" | tee -a "$RESULT_ROOT/logs/resume.log"
$PY -m flower_classification.grid_search \
  --data-root . \
  --output-root "$RESULT_ROOT/core_grid" \
  --models resnet18,resnet34,resnet18_se,resnet34_se,resnet18_cbam,resnet34_cbam \
  --scratch-models resnet18,resnet34 \
  --epochs 200 \
  --patience 10 \
  --num-workers 8 \
  --learning-rates 0.001,0.0003 \
  --weight-decays 0.0001,0.00001 \
  --batch-sizes 32,64 \
  --label-smoothing 0,0.1 \
  2>&1 | tee -a "$RESULT_ROOT/logs/core_grid_resume.log"
echo "===== TRANSFORMER GRID RESUME $(date '+%F %T') =====" | tee -a "$RESULT_ROOT/logs/resume.log"
$PY -m flower_classification.grid_search \
  --data-root . \
  --output-root "$RESULT_ROOT/transformer_grid" \
  --models vit_b_16,swin_t \
  --scratch-models= \
  --epochs 200 \
  --patience 10 \
  --num-workers 8 \
  --learning-rates 0.0003,0.0001 \
  --weight-decays 0.0001 \
  --batch-sizes 16,32 \
  --label-smoothing 0,0.1 \
  2>&1 | tee -a "$RESULT_ROOT/logs/transformer_grid_resume.log"
echo "===== GRID RESUME DONE $(date '+%F %T') =====" | tee -a "$RESULT_ROOT/logs/resume.log"
