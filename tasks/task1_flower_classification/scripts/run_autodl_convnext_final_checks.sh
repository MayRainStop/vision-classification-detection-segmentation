#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/flower_classification_task1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

PY=/root/miniconda3/bin/python
RESULT_ROOT=experiment_results
OUT_ROOT="$RESULT_ROOT/convnext_final_checks"
LOG_DIR="$RESULT_ROOT/logs"
LOG_FILE="$LOG_DIR/convnext_final_checks.log"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "===== CONVNEXT FINAL CHECKS START $(date '+%F %T') =====" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE" || true

run_trial() {
  local image_size="$1"
  local seed="$2"
  local run_name="$3"

  if [ -f "$OUT_ROOT/$run_name/summary.json" ]; then
    echo "===== skip completed $run_name =====" | tee -a "$LOG_FILE"
    return
  fi

  echo "===== run $run_name image_size=$image_size seed=$seed $(date '+%F %T') =====" | tee -a "$LOG_FILE"
  "$PY" -m flower_classification.train \
    --data-root . \
    --model convnext_tiny \
    --epochs 200 \
    --batch-size 16 \
    --lr 0.0001 \
    --weight-decay 0.00001 \
    --label-smoothing 0.1 \
    --early-stopping-patience 15 \
    --image-size "$image_size" \
    --num-workers 8 \
    --output-dir "$OUT_ROOT" \
    --run-name "$run_name" \
    --seed "$seed" \
    2>&1 | tee -a "$LOG_FILE"
}

# Extend the strongest 224px ConvNeXt-Tiny configuration from 3 seeds to 10 seeds.
# Seeds 7, 123, and 2026 already exist in experiment_results/supplementary_repeats.
for seed in 42 99 314 512 1009 2024 4096; do
  run_trial 224 "$seed" "convnext_tiny_224_best_seed${seed}_lr0p0001_wd1e-05_bs16_ls0p1"
done

# Check whether higher input resolution improves the same stable ConvNeXt-Tiny setup.
for seed in 7 123 2026; do
  run_trial 384 "$seed" "convnext_tiny_384_best_seed${seed}_lr0p0001_wd1e-05_bs16_ls0p1"
done

"$PY" - <<'PY'
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

out_root = Path("experiment_results/convnext_final_checks")
supp_root = Path("experiment_results/supplementary_repeats")

rows = []


def add_summary(summary_path: Path, group: str, source: str, image_size: int) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    args = summary["args"]
    rows.append(
        {
            "group": group,
            "source": source,
            "model": args["model"],
            "run_name": args["run_name"],
            "seed": args["seed"],
            "image_size": image_size,
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


existing_224 = {
    7: "convnext_tiny_lr0p0001_wd1e-05_bs16_seed7_ls0p1",
    123: "convnext_tiny_lr0p0001_wd1e-05_bs16_seed123_ls0p1",
    2026: "convnext_tiny_lr0p0001_wd1e-05_bs16_seed2026_ls0p1",
}
for seed, run_name in existing_224.items():
    path = supp_root / run_name / "summary.json"
    if path.exists():
        add_summary(path, "convnext_tiny_224_best10", "supplementary_repeats", 224)

for summary_path in sorted(out_root.glob("*/summary.json")):
    run_name = summary_path.parent.name
    if run_name.startswith("convnext_tiny_224_best_"):
        add_summary(summary_path, "convnext_tiny_224_best10", "convnext_final_checks", 224)
    elif run_name.startswith("convnext_tiny_384_best_"):
        add_summary(summary_path, "convnext_tiny_384_best3", "convnext_final_checks", 384)

rows.sort(key=lambda row: (row["group"], int(row["seed"])))
summary_csv = out_root / "convnext_final_checks_summary.csv"
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
            "image_size": first["image_size"],
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

stats_csv = out_root / "convnext_final_checks_stats.csv"
if stats_rows:
    with stats_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)
    (out_root / "convnext_final_checks_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")

print(f"wrote {summary_csv} rows={len(rows)}")
print(f"wrote {stats_csv} groups={len(stats_rows)}")
PY

echo "===== CONVNEXT FINAL CHECKS DONE $(date '+%F %T') =====" | tee -a "$LOG_FILE"
