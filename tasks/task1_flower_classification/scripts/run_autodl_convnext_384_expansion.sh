#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/flower_classification_task1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

PY=/root/miniconda3/bin/python
RESULT_ROOT=experiment_results
OUT_ROOT="$RESULT_ROOT/convnext_384_expansion"
FINAL_ROOT="$RESULT_ROOT/convnext_final_checks"
LOG_DIR="$RESULT_ROOT/logs"
LOG_FILE="$LOG_DIR/convnext_384_expansion.log"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "===== CONVNEXT 384 EXPANSION START $(date '+%F %T') =====" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE" || true

run_trial() {
  local lr="$1"
  local wd="$2"
  local seed="$3"
  local run_name="$4"

  if [ -f "$OUT_ROOT/$run_name/summary.json" ]; then
    echo "===== skip completed $run_name =====" | tee -a "$LOG_FILE"
    return
  fi

  echo "===== run $run_name lr=$lr wd=$wd seed=$seed $(date '+%F %T') =====" | tee -a "$LOG_FILE"
  "$PY" -m flower_classification.train \
    --data-root . \
    --model convnext_tiny \
    --epochs 200 \
    --batch-size 16 \
    --lr "$lr" \
    --weight-decay "$wd" \
    --label-smoothing 0.1 \
    --early-stopping-patience 15 \
    --image-size 384 \
    --num-workers 8 \
    --output-dir "$OUT_ROOT" \
    --run-name "$run_name" \
    --seed "$seed" \
    2>&1 | tee -a "$LOG_FILE"
}

# Extend the current best 384px ConvNeXt-Tiny setting from 3 seeds to 10 seeds.
# Seeds 7, 123, and 2026 already exist in experiment_results/convnext_final_checks.
for seed in 42 99 314 512 1009 2024 4096; do
  run_trial 0.0001 0.00001 "$seed" "convnext_tiny_384_best_seed${seed}_lr0p0001_wd1e-05_bs16_ls0p1"
done

# Local 384px tuning grid, fixed batch size and label smoothing.
# The baseline lr=0.0001/wd=0.00001/seed=7 already exists in convnext_final_checks.
run_trial 0.0003 0.0001 7 "convnext_tiny_384_tune_seed7_lr0p0003_wd0p0001_bs16_ls0p1"
run_trial 0.0003 0.00001 7 "convnext_tiny_384_tune_seed7_lr0p0003_wd1e-05_bs16_ls0p1"
run_trial 0.0001 0.0001 7 "convnext_tiny_384_tune_seed7_lr0p0001_wd0p0001_bs16_ls0p1"
run_trial 0.00005 0.0001 7 "convnext_tiny_384_tune_seed7_lr5e-05_wd0p0001_bs16_ls0p1"
run_trial 0.00005 0.00001 7 "convnext_tiny_384_tune_seed7_lr5e-05_wd1e-05_bs16_ls0p1"

TOP_CHALLENGER=$("$PY" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

out_root = Path("experiment_results/convnext_384_expansion")
configs = [
    ("0.0003", "0.0001", "lr0p0003_wd0p0001", "convnext_tiny_384_tune_seed7_lr0p0003_wd0p0001_bs16_ls0p1"),
    ("0.0003", "0.00001", "lr0p0003_wd1e-05", "convnext_tiny_384_tune_seed7_lr0p0003_wd1e-05_bs16_ls0p1"),
    ("0.0001", "0.0001", "lr0p0001_wd0p0001", "convnext_tiny_384_tune_seed7_lr0p0001_wd0p0001_bs16_ls0p1"),
    ("0.00005", "0.0001", "lr5e-05_wd0p0001", "convnext_tiny_384_tune_seed7_lr5e-05_wd0p0001_bs16_ls0p1"),
    ("0.00005", "0.00001", "lr5e-05_wd1e-05", "convnext_tiny_384_tune_seed7_lr5e-05_wd1e-05_bs16_ls0p1"),
]

rows = []
for lr, wd, suffix, run_name in configs:
    path = out_root / run_name / "summary.json"
    if not path.exists():
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows.append((float(summary["best_val_acc"]), float(summary["test_acc"]), lr, wd, suffix))

if not rows:
    raise SystemExit("no tuning challenger summaries found")

rows.sort(reverse=True)
_, _, lr, wd, suffix = rows[0]
print(f"{lr} {wd} {suffix}")
PY
)
read -r TOP_LR TOP_WD TOP_SUFFIX <<< "$TOP_CHALLENGER"
echo "===== top non-baseline challenger lr=$TOP_LR wd=$TOP_WD suffix=$TOP_SUFFIX =====" | tee -a "$LOG_FILE"

for seed in 123 2026; do
  run_trial "$TOP_LR" "$TOP_WD" "$seed" "convnext_tiny_384_tune_top_${TOP_SUFFIX}_seed${seed}_bs16_ls0p1"
done

"$PY" - <<'PY'
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

out_root = Path("experiment_results/convnext_384_expansion")
final_root = Path("experiment_results/convnext_final_checks")

all_rows: list[dict[str, object]] = []
tuning_rows: list[dict[str, object]] = []


def read_summary(path: Path, collection: str, group: str, source: str) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    args = summary["args"]
    return {
        "collection": collection,
        "group": group,
        "source": source,
        "model": args["model"],
        "run_name": args["run_name"],
        "seed": args["seed"],
        "image_size": args["image_size"],
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


baseline_existing = {
    7: "convnext_tiny_384_best_seed7_lr0p0001_wd1e-05_bs16_ls0p1",
    123: "convnext_tiny_384_best_seed123_lr0p0001_wd1e-05_bs16_ls0p1",
    2026: "convnext_tiny_384_best_seed2026_lr0p0001_wd1e-05_bs16_ls0p1",
}
for run_name in baseline_existing.values():
    path = final_root / run_name / "summary.json"
    if path.exists():
        row = read_summary(path, "best10", "convnext_tiny_384_best10", "convnext_final_checks")
        all_rows.append(row)

for seed in [42, 99, 314, 512, 1009, 2024, 4096]:
    run_name = f"convnext_tiny_384_best_seed{seed}_lr0p0001_wd1e-05_bs16_ls0p1"
    path = out_root / run_name / "summary.json"
    if path.exists():
        row = read_summary(path, "best10", "convnext_tiny_384_best10", "convnext_384_expansion")
        all_rows.append(row)

tuning_configs = [
    ("lr0p0003_wd0p0001", "convnext_tiny_384_tune_seed7_lr0p0003_wd0p0001_bs16_ls0p1", out_root),
    ("lr0p0003_wd1e-05", "convnext_tiny_384_tune_seed7_lr0p0003_wd1e-05_bs16_ls0p1", out_root),
    ("lr0p0001_wd0p0001", "convnext_tiny_384_tune_seed7_lr0p0001_wd0p0001_bs16_ls0p1", out_root),
    ("lr0p0001_wd1e-05", "convnext_tiny_384_best_seed7_lr0p0001_wd1e-05_bs16_ls0p1", final_root),
    ("lr5e-05_wd0p0001", "convnext_tiny_384_tune_seed7_lr5e-05_wd0p0001_bs16_ls0p1", out_root),
    ("lr5e-05_wd1e-05", "convnext_tiny_384_tune_seed7_lr5e-05_wd1e-05_bs16_ls0p1", out_root),
]

top_nonbaseline: str | None = None
for group, run_name, root in tuning_configs:
    path = root / run_name / "summary.json"
    if path.exists():
        source = "convnext_final_checks" if root == final_root else "convnext_384_expansion"
        row = read_summary(path, "tuning_seed7", f"convnext_tiny_384_tune_seed7_{group}", source)
        all_rows.append(row)
        tuning_rows.append(row)

nonbaseline_tuning = [row for row in tuning_rows if str(row["group"]) != "convnext_tiny_384_tune_seed7_lr0p0001_wd1e-05"]
if nonbaseline_tuning:
    nonbaseline_tuning.sort(key=lambda row: (float(row["best_val_acc"]), float(row["test_acc"])), reverse=True)
    top_nonbaseline = str(nonbaseline_tuning[0]["group"]).removeprefix("convnext_tiny_384_tune_seed7_")

if top_nonbaseline is not None:
    seed7_group = f"convnext_tiny_384_tune_seed7_{top_nonbaseline}"
    for row in tuning_rows:
        if row["group"] == seed7_group:
            repeated = dict(row)
            repeated["collection"] = "top_challenger3"
            repeated["group"] = f"convnext_tiny_384_top_challenger3_{top_nonbaseline}"
            all_rows.append(repeated)

    for seed in [123, 2026]:
        run_name = f"convnext_tiny_384_tune_top_{top_nonbaseline}_seed{seed}_bs16_ls0p1"
        path = out_root / run_name / "summary.json"
        if path.exists():
            row = read_summary(path, "top_challenger3", f"convnext_tiny_384_top_challenger3_{top_nonbaseline}", "convnext_384_expansion")
            all_rows.append(row)

all_rows.sort(key=lambda row: (str(row["collection"]), str(row["group"]), int(row["seed"])))
summary_csv = out_root / "convnext_384_expansion_summary.csv"
if all_rows:
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

tuning_rows.sort(key=lambda row: (float(row["best_val_acc"]), float(row["test_acc"])), reverse=True)
tuning_csv = out_root / "convnext_384_tuning_single_seed.csv"
if tuning_rows:
    with tuning_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(tuning_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tuning_rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


stats_rows = []
by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
for row in all_rows:
    if row["collection"] in {"best10", "top_challenger3"}:
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

stats_csv = out_root / "convnext_384_expansion_stats.csv"
if stats_rows:
    with stats_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)
    (out_root / "convnext_384_expansion_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")

metadata = {
    "top_nonbaseline_challenger": top_nonbaseline,
    "num_summary_rows": len(all_rows),
    "num_tuning_rows": len(tuning_rows),
    "num_stats_rows": len(stats_rows),
}
(out_root / "convnext_384_expansion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"wrote {summary_csv} rows={len(all_rows)}")
print(f"wrote {tuning_csv} rows={len(tuning_rows)}")
print(f"wrote {stats_csv} groups={len(stats_rows)}")
print(json.dumps(metadata, indent=2))
PY

echo "===== CONVNEXT 384 EXPANSION DONE $(date '+%F %T') =====" | tee -a "$LOG_FILE"
