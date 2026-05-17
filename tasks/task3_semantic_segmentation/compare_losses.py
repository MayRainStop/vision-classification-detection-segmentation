from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.segmentation.dataset import TASK_METADATA, load_split_ids


LOSS_CONFIGS = {
    "ce": {"label": "Cross-Entropy", "loss_mode": "ce"},
    "dice": {"label": "Dice", "loss_mode": "dice"},
    "combo": {"label": "Cross-Entropy + Dice", "loss_mode": "combo"},
}

PALETTE = np.array(
    [
        [70, 130, 180],
        [34, 139, 34],
        [128, 64, 128],
        [124, 252, 0],
        [30, 144, 255],
        [184, 134, 11],
        [139, 137, 137],
        [220, 20, 60],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare CE / Dice / CE+Dice segmentation experiments.")
    parser.add_argument("--task", type=str, default="regions", choices=sorted(TASK_METADATA))
    parser.add_argument("--data-root", type=str, default="iccv09Data")
    parser.add_argument("--output-root", type=str, default="runs/loss_comparison")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("W", "H"))
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def build_palette(num_classes: int) -> np.ndarray:
    if num_classes <= len(PALETTE):
        return PALETTE[:num_classes]
    rng = np.random.default_rng(42)
    extra = rng.integers(0, 255, size=(num_classes - len(PALETTE), 3), dtype=np.uint8)
    return np.concatenate([PALETTE, extra], axis=0)


def run_training(args: argparse.Namespace, config_name: str, output_dir: Path) -> None:
    command = [
        sys.executable,
        "train.py",
        "--task",
        args.task,
        "--data-root",
        args.data_root,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size[0]),
        str(args.image_size[1]),
        "--base-channels",
        str(args.base_channels),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--loss-mode",
        LOSS_CONFIGS[config_name]["loss_mode"],
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True)


def load_history(run_dir: Path) -> list[dict[str, float]]:
    with (run_dir / "history.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def plot_comparison_curves(histories: dict[str, list[dict[str, float]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for config_name, history in histories.items():
        label = LOSS_CONFIGS[config_name]["label"]
        epochs = [record["epoch"] for record in history]
        axes[0].plot(epochs, [record["val_loss"] for record in history], label=label)
        axes[1].plot(epochs, [record["val_miou"] for record in history], label=label)

    axes[0].set_title("Validation Loss Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Validation mIoU Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "miou_loss_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary(histories: dict[str, list[dict[str, float]]], output_dir: Path) -> None:
    summary = {}
    for config_name, history in histories.items():
        best_record = max(history, key=lambda record: record["val_miou"])
        summary[config_name] = {
            "label": LOSS_CONFIGS[config_name]["label"],
            "best_epoch": best_record["epoch"],
            "best_val_miou": best_record["val_miou"],
            "best_val_pixel_acc": best_record["val_pixel_acc"],
            "best_val_loss": best_record["val_loss"],
            "val_per_class_iou": best_record.get("val_per_class_iou", []),
        }

    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def save_confusion_grid(args: argparse.Namespace, output_dir: Path) -> None:
    metadata = TASK_METADATA[args.task]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for axis, config_name in zip(axes, LOSS_CONFIGS):
        hist = torch.load(output_dir / config_name / "best_confusion_matrix.pt", map_location="cpu").float()
        normalized = (hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)).numpy()
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(LOSS_CONFIGS[config_name]["label"])
        axis.set_xticks(range(len(metadata.class_names)))
        axis.set_yticks(range(len(metadata.class_names)))
        axis.set_xticklabels(metadata.class_names, rotation=45, ha="right", fontsize=8)
        axis.set_yticklabels(metadata.class_names, fontsize=8)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Ground Truth")

    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(output_dir / "confusion_matrix_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prediction_grid(args: argparse.Namespace, output_dir: Path) -> None:
    image_dir = Path(args.data_root) / "images"
    label_dir = Path(args.data_root) / "labels"
    _, val_ids = load_split_ids(image_dir, val_ratio=0.2, seed=args.seed)
    sample_id = args.sample_id or val_ids[0]
    image_path = image_dir / f"{sample_id}.jpg"
    label_path = label_dir / f"{sample_id}.{TASK_METADATA[args.task].label_suffix}.txt"

    image = Image.open(image_path).convert("RGB")
    label = np.loadtxt(label_path, dtype=np.int64)
    label[label < 0] = 0
    palette = build_palette(TASK_METADATA[args.task].num_classes)
    gt_color = palette[label]

    generated = []
    for config_name in LOSS_CONFIGS:
        prediction_path = output_dir / f"{sample_id}_{config_name}.png"
        subprocess.run(
            [
                sys.executable,
                "predict.py",
                "--checkpoint",
                str(output_dir / config_name / "best.pt"),
                "--image",
                str(image_path),
                "--output",
                str(prediction_path),
                "--device",
                args.device,
            ],
            check=True,
        )
        generated.append((LOSS_CONFIGS[config_name]["label"], Image.open(prediction_path).convert("RGB")))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth")

    for idx, (label_name, pred_image) in enumerate(generated, start=2):
        axes[idx].imshow(pred_image)
        axes[idx].set_title(label_name)

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{sample_id}_prediction_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories: dict[str, list[dict[str, float]]] = {}

    for config_name in LOSS_CONFIGS:
        run_dir = output_dir / config_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_train:
            run_training(args, config_name, run_dir)
        histories[config_name] = load_history(run_dir)

    plot_comparison_curves(histories, output_dir)
    save_summary(histories, output_dir)
    save_confusion_grid(args, output_dir)
    save_prediction_grid(args, output_dir)


if __name__ == "__main__":
    main()
