from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor the extended CNN grid, summarize results, and run seed repeats for the best trial."
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--grid-root", type=Path, default=Path("experiment_results/extended_cnn_grid"))
    parser.add_argument("--report-root", type=Path, default=Path("experiment_results/extended_cnn_report"))
    parser.add_argument("--seed-root", type=Path, default=Path("experiment_results/extended_cnn_seed_runs"))
    parser.add_argument("--pid-file", type=Path, default=Path("extended_cnn_master.pid"))
    parser.add_argument("--expected-trials", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--resume-attempts", type=int, default=2)
    parser.add_argument("--seeds", default="7,123,2026")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    args.seed_root.mkdir(parents=True, exist_ok=True)

    resume_attempts = 0
    while True:
        completed = count_summaries(args.grid_root)
        running = process_running(read_pid(args.pid_file))
        print(f"monitor completed={completed}/{args.expected_trials} running={running}", flush=True)
        if completed > 0:
            summarize_grid(args.grid_root, args.report_root)

        if completed >= args.expected_trials:
            break

        if not running:
            if resume_attempts < args.resume_attempts:
                resume_attempts += 1
                print(f"grid not running; resume attempt {resume_attempts}/{args.resume_attempts}", flush=True)
                start_extended_grid(args.pid_file)
            else:
                summarize_grid(args.grid_root, args.report_root)
                raise SystemExit(
                    "Extended CNN grid stopped before all expected summaries; "
                    "not running seed repeats from incomplete results."
                )

        time.sleep(args.poll_seconds)

    rows = summarize_grid(args.grid_root, args.report_root)
    if not rows:
        raise SystemExit("No completed extended CNN summaries found.")

    best = rows[0]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    run_seed_repeats(best, seeds, args)
    summarize_seed_runs(args.seed_root, args.report_root)


def count_summaries(root: Path) -> int:
    return sum(1 for _ in root.glob("*/summary.json"))


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def process_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_extended_grid(pid_file: Path) -> None:
    cmd = ["bash", "scripts/run_autodl_extended_cnn.sh"]
    log_path = Path("experiment_results/logs/extended_cnn_nohup.out")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    pid_file.write_text(str(process.pid), encoding="utf-8")
    print(f"resumed extended grid pid={process.pid}", flush=True)


def summarize_grid(root: Path, report_root: Path) -> list[dict[str, str | int | float | bool]]:
    rows = []
    for summary_path in root.glob("*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        args = summary["args"]
        rows.append(
            {
                "model": args["model"],
                "run_name": args["run_name"],
                "pretrained": not bool(args.get("no_pretrained", False)),
                "learning_rate": args["lr"],
                "weight_decay": args["weight_decay"],
                "batch_size": args["batch_size"],
                "label_smoothing": args["label_smoothing"],
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "best_val_acc": summary["best_val_acc"],
                "test_acc": summary["test_acc"],
                "test_loss": summary["test_loss"],
                "summary_path": str(summary_path),
            }
        )

    rows.sort(key=lambda row: float(row["best_val_acc"]), reverse=True)
    write_csv(report_root / "extended_cnn_all_sorted.csv", rows)

    best_by_model = {}
    for row in rows:
        model = str(row["model"])
        if model not in best_by_model:
            best_by_model[model] = row
    write_csv(report_root / "extended_cnn_best_by_model.csv", list(best_by_model.values()))

    if rows:
        (report_root / "best_extended_cnn.json").write_text(json.dumps(rows[0], indent=2), encoding="utf-8")
        print(
            "best extended CNN "
            f"{rows[0]['model']} val={float(rows[0]['best_val_acc']) * 100:.2f}% "
            f"test={float(rows[0]['test_acc']) * 100:.2f}% run={rows[0]['run_name']}",
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}", flush=True)


def run_seed_repeats(best: dict[str, str | int | float | bool], seeds: list[int], args: argparse.Namespace) -> None:
    for seed in seeds:
        run_name = format_seed_run_name(best, seed)
        run_dir = args.seed_root / run_name
        if (run_dir / "summary.json").exists():
            print(f"skipping completed seed repeat {run_name}", flush=True)
            continue

        cmd = [
            sys.executable,
            "-m",
            "flower_classification.train",
            "--data-root",
            str(args.data_root),
            "--model",
            str(best["model"]),
            "--epochs",
            "200",
            "--batch-size",
            str(best["batch_size"]),
            "--lr",
            str(best["learning_rate"]),
            "--weight-decay",
            str(best["weight_decay"]),
            "--label-smoothing",
            str(best["label_smoothing"]),
            "--early-stopping-patience",
            "15",
            "--image-size",
            "224",
            "--num-workers",
            str(args.num_workers),
            "--output-dir",
            str(args.seed_root),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
        ]
        if not bool(best["pretrained"]):
            cmd.append("--no-pretrained")

        print("running seed repeat " + " ".join(cmd), flush=True)
        process = subprocess.run(cmd, text=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"Seed repeat failed for {run_name} with return code {process.returncode}")


def format_seed_run_name(best: dict[str, str | int | float | bool], seed: int) -> str:
    return (
        f"{best['model']}_best_seed{seed}_"
        f"lr{format_number(float(best['learning_rate']))}_"
        f"wd{format_number(float(best['weight_decay']))}_"
        f"bs{best['batch_size']}_ls{format_number(float(best['label_smoothing']))}"
    )


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def summarize_seed_runs(seed_root: Path, report_root: Path) -> None:
    rows = []
    for summary_path in seed_root.glob("*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        args = summary["args"]
        rows.append(
            {
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

    rows.sort(key=lambda row: int(row["seed"]))
    write_csv(report_root / "best_cnn_seed_repeats.csv", rows)
    if not rows:
        return

    val_scores = [float(row["best_val_acc"]) for row in rows]
    test_scores = [float(row["test_acc"]) for row in rows]
    stats = {
        "num_runs": len(rows),
        "val_acc_mean": mean(val_scores),
        "val_acc_std": sample_std(val_scores),
        "test_acc_mean": mean(test_scores),
        "test_acc_std": sample_std(test_scores),
    }
    (report_root / "best_cnn_seed_repeat_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"seed repeats test mean={stats['test_acc_mean'] * 100:.2f}% "
        f"std={stats['test_acc_std'] * 100:.2f}%",
        flush=True,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    main()
