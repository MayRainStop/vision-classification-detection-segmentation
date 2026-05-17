#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/flower_classification_task1 || exit 1
CACHE=/root/.cache/torch/hub/checkpoints
mkdir -p "$CACHE" experiment_results/logs
LOG=experiment_results/logs/prefetch_extended_weights.log
exec > >(tee -a "$LOG") 2>&1

echo "===== PREFETCH START $(date '+%F %T') ====="

fetch() {
  local url="$1"
  local file="$2"
  local dest="$CACHE/$file"
  local tmp="$dest.part"
  if [ -s "$dest" ]; then
    echo "skip existing $file $(stat -c%s "$dest") bytes"
    return 0
  fi
  local partial
  partial=$(ls "$dest".*.partial 2>/dev/null | head -1 || true)
  if [ -n "$partial" ] && [ ! -s "$tmp" ]; then
    echo "reuse partial $partial -> $tmp"
    mv "$partial" "$tmp"
  fi
  local attempt=1
  while [ ! -s "$dest" ]; do
    echo "download attempt $attempt $file $(date '+%F %T')"
    curl -L --fail --retry 8 --retry-delay 5 --connect-timeout 20 --speed-limit 2048 --speed-time 60 -C - -o "$tmp" "$url"
    code=$?
    if [ "$code" -eq 0 ]; then
      mv "$tmp" "$dest"
      echo "downloaded $file $(stat -c%s "$dest") bytes"
      return 0
    fi
    echo "curl failed code=$code for $file"
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 20 ]; then
      echo "failed to download $file after retries"
      exit 1
    fi
    sleep 10
  done
}

fetch https://download.pytorch.org/models/resnet101-cd907fc2.pth resnet101-cd907fc2.pth
fetch https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth efficientnet_b0_rwightman-7f5810bc.pth
fetch https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth efficientnet_b3_rwightman-b3899882.pth
fetch https://download.pytorch.org/models/convnext_tiny-983f1562.pth convnext_tiny-983f1562.pth

echo "===== PREFETCH DONE $(date '+%F %T') ====="
echo "===== RESUME EXTENDED CNN $(date '+%F %T') ====="
nohup bash scripts/run_autodl_extended_cnn.sh > experiment_results/logs/extended_cnn_nohup.out 2>&1 &
echo $! > extended_cnn_master.pid
echo "resumed pid $(cat extended_cnn_master.pid)"
