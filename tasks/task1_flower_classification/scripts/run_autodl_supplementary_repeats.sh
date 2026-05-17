#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/flower_classification_task1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

PY=/root/miniconda3/bin/python
RESULT_ROOT=experiment_results
OUT_ROOT="$RESULT_ROOT/supplementary_repeats"
LOG_DIR="$RESULT_ROOT/logs"
LOG_FILE="$LOG_DIR/supplementary_repeats.log"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "===== SUPPLEMENTARY REPEATS START $(date '+%F %T') =====" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE" || true

run_trial() {
  local model="$1"
  local lr="$2"
  local wd="$3"
  local bs="$4"
  local ls="$5"
  local patience="$6"
  local seed="$7"
  local run_name="$8"

  if [ -f "$OUT_ROOT/$run_name/summary.json" ]; then
    echo "===== skip completed $run_name =====" | tee -a "$LOG_FILE"
    return
  fi

  echo "===== run $run_name $(date '+%F %T') =====" | tee -a "$LOG_FILE"
  "$PY" -m flower_classification.train \
    --data-root . \
    --model "$model" \
    --epochs 200 \
    --batch-size "$bs" \
    --lr "$lr" \
    --weight-decay "$wd" \
    --label-smoothing "$ls" \
    --early-stopping-patience "$patience" \
    --image-size 224 \
    --num-workers 8 \
    --output-dir "$OUT_ROOT" \
    --run-name "$run_name" \
    --seed "$seed" \
    2>&1 | tee -a "$LOG_FILE"
}

for seed in 7 123 2026; do
  run_trial convnext_tiny 0.0003 0.00001 32 0.1 15 "$seed" "convnext_tiny_wd1e-05_seed${seed}_lr0p0003_wd1e-05_bs32_ls0p1"
  run_trial convnext_tiny 0.0001 0.00001 16 0.1 15 "$seed" "convnext_tiny_lr0p0001_wd1e-05_bs16_seed${seed}_ls0p1"
  run_trial swin_t 0.0001 0.0001 16 0.1 10 "$seed" "swin_t_best_bs16_seed${seed}_lr0p0001_wd0p0001_ls0p1"
  run_trial swin_t 0.0001 0.0001 32 0.1 10 "$seed" "swin_t_best_bs32_seed${seed}_lr0p0001_wd0p0001_ls0p1"
done

"$PY" - <<'PY'
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

root = Path("experiment_results/supplementary_repeats")
rows = []
for summary_path in root.glob("*/summary.json"):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    args = summary["args"]
    rows.append(
        {
            "group": args["run_name"].split("_seed", 1)[0],
            "model": args["model"],
            "run_name": args["run_name"],
            "seed": args["seed"],
            "learning_rate": args["lr"],
            "weight_decay": args["weight_decay"],
            "batch_size": args["batch_size"],
            "label_smoothing": args["label_smoothing"],
            "best_epoch": summary["best_epoch"],
            "epochs_completed": summary["epochs_completed"],
            "best_val_acc": summary["best_val_acc"],
            "test_acc": summary["test_acc"],
            "test_loss": summary["test_loss"],
        }
    )

rows.sort(key=lambda row: (row["group"], int(row["seed"])))
summary_csv = root / "supplementary_repeats_summary.csv"
if rows:
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))

stats_rows = []
by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
for row in rows:
    by_group[str(row["group"])].append(row)

for group, group_rows in sorted(by_group.items()):
    val_scores = [float(row["best_val_acc"]) for row in group_rows]
    test_scores = [float(row["test_acc"]) for row in group_rows]
    first = group_rows[0]
    stats_rows.append(
        {
            "group": group,
            "model": first["model"],
            "num_runs": len(group_rows),
            "learning_rate": first["learning_rate"],
            "weight_decay": first["weight_decay"],
            "batch_size": first["batch_size"],
            "label_smoothing": first["label_smoothing"],
            "val_acc_mean": mean(val_scores),
            "val_acc_std": sample_std(val_scores),
            "test_acc_mean": mean(test_scores),
            "test_acc_std": sample_std(test_scores),
        }
    )

stats_csv = root / "supplementary_repeats_stats.csv"
if stats_rows:
    with stats_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)
    (root / "supplementary_repeats_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")

print(f"wrote {summary_csv} rows={len(rows)}")
print(f"wrote {stats_csv} groups={len(stats_rows)}")
PY

echo "===== SUPPLEMENTARY REPEATS DONE $(date '+%F %T') =====" | tee -a "$LOG_FILE"
