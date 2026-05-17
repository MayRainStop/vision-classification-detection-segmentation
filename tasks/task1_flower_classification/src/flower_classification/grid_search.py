from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridValues:
    learning_rates: list[float]
    weight_decays: list[float]
    batch_sizes: list[int]
    label_smoothing: list[float]


@dataclass(frozen=True)
class TrialConfig:
    model: str
    pretrained: bool
    learning_rate: float
    weight_decay: float
    batch_size: int
    label_smoothing: float

    @property
    def name(self) -> str:
        return format_trial_name(
            self.model,
            self.pretrained,
            self.learning_rate,
            self.weight_decay,
            self.batch_size,
            self.label_smoothing,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Oxford 102 Flowers hyperparameter grid search.")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("grid_runs"))
    parser.add_argument("--models", default="resnet18,resnet18_se,resnet18_cbam")
    parser.add_argument("--scratch-models", default="resnet18")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rates", default="0.001,0.0003")
    parser.add_argument("--weight-decays", default="0.0001,0.00001")
    parser.add_argument("--batch-sizes", default="32,64")
    parser.add_argument("--label-smoothing", default="0,0.1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rerun", action="store_true", help="Run trials even if summary.json already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned trials without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = GridValues(
        learning_rates=_parse_floats(args.learning_rates),
        weight_decays=_parse_floats(args.weight_decays),
        batch_sizes=_parse_ints(args.batch_sizes),
        label_smoothing=_parse_floats(args.label_smoothing),
    )
    trials = build_trial_configs(_parse_strings(args.models), _parse_strings(args.scratch_models), values)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for trial in trials:
            print(trial.name)
        return

    rows = []
    for index, trial in enumerate(trials, start=1):
        print(f"===== trial {index}/{len(trials)} {trial.name} =====", flush=True)
        rows.append(run_trial(args, trial))
        write_grid_summary(args.output_root / "grid_summary.csv", rows)


def build_trial_configs(models: list[str], scratch_models: list[str], values: GridValues) -> list[TrialConfig]:
    trials: list[TrialConfig] = []
    for model in models:
        trials.extend(_configs_for_model(model, True, values))
    for model in scratch_models:
        trials.extend(_configs_for_model(model, False, values))
    return trials


def format_trial_name(
    model: str,
    pretrained: bool,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    label_smoothing: float,
) -> str:
    mode = "pretrained" if pretrained else "scratch"
    return (
        f"{model}_{mode}_lr{_format_number(learning_rate)}_wd{_format_number(weight_decay)}_"
        f"bs{batch_size}_ls{_format_number(label_smoothing)}"
    )


def run_trial(args: argparse.Namespace, trial: TrialConfig) -> dict[str, str | int | float]:
    run_dir = args.output_root / trial.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "train.log"

    if summary_path.exists() and not args.rerun:
        row = _row_from_summary(trial, summary_path, "skipped")
        print(f"skipping completed trial {trial.name}", flush=True)
        return row

    cmd = [
        sys.executable,
        "-m",
        "flower_classification.train",
        "--data-root",
        str(args.data_root),
        "--model",
        trial.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(trial.batch_size),
        "--lr",
        str(trial.learning_rate),
        "--weight-decay",
        str(trial.weight_decay),
        "--label-smoothing",
        str(trial.label_smoothing),
        "--early-stopping-patience",
        str(args.patience),
        "--image-size",
        str(args.image_size),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(args.output_root),
        "--run-name",
        trial.name,
        "--seed",
        str(args.seed),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if not trial.pretrained:
        cmd.append("--no-pretrained")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)

    if process.returncode == 0 and summary_path.exists():
        return _row_from_summary(trial, summary_path, "completed")

    return _base_row(trial, status="failed", return_code=process.returncode)


def write_grid_summary(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fieldnames = [
        "status",
        "model",
        "pretrained",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "label_smoothing",
        "epochs_completed",
        "best_epoch",
        "best_val_acc",
        "test_loss",
        "test_acc",
        "early_stopped",
        "return_code",
        "run_name",
    ]
    sorted_rows = sorted(rows, key=lambda row: float(row.get("best_val_acc") or -1), reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)


def _configs_for_model(model: str, pretrained: bool, values: GridValues) -> list[TrialConfig]:
    configs = []
    for lr, wd, batch_size, smoothing in itertools.product(
        values.learning_rates,
        values.weight_decays,
        values.batch_sizes,
        values.label_smoothing,
    ):
        configs.append(TrialConfig(model, pretrained, lr, wd, batch_size, smoothing))
    return configs


def _row_from_summary(trial: TrialConfig, summary_path: Path, status: str) -> dict[str, str | int | float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = _base_row(trial, status=status, return_code=0)
    row.update(
        {
            "epochs_completed": summary["epochs_completed"],
            "best_epoch": summary["best_epoch"],
            "best_val_acc": summary["best_val_acc"],
            "test_loss": summary["test_loss"],
            "test_acc": summary["test_acc"],
            "early_stopped": summary["early_stopped"],
        }
    )
    return row


def _base_row(trial: TrialConfig, status: str, return_code: int) -> dict[str, str | int | float]:
    return {
        "status": status,
        "model": trial.model,
        "pretrained": trial.pretrained,
        "learning_rate": trial.learning_rate,
        "weight_decay": trial.weight_decay,
        "batch_size": trial.batch_size,
        "label_smoothing": trial.label_smoothing,
        "epochs_completed": "",
        "best_epoch": "",
        "best_val_acc": "",
        "test_loss": "",
        "test_acc": "",
        "early_stopped": "",
        "return_code": return_code,
        "run_name": trial.name,
    }


def _format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _parse_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(item) for item in _parse_strings(raw)]


def _parse_ints(raw: str) -> list[int]:
    return [int(item) for item in _parse_strings(raw)]


if __name__ == "__main__":
    main()
