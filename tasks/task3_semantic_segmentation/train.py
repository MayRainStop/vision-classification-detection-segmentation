from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.dataset import CLASS_NAMES, IGNORE_INDEX, StanfordBackgroundDataset, TASK_METADATA
from src.segmentation.losses import build_loss
from src.segmentation.metrics import confusion_matrix, summarize_metrics
from src.segmentation.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a handwritten U-Net on Stanford Background Dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("iccv09Data_prepared"))
    parser.add_argument("--task", type=str, default="regions", choices=("regions",))
    parser.add_argument("--loss-mode", type=str, default="combo", choices=("ce", "dice", "combo"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(320, 240), metavar=("W", "H"))
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=False, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegmentationMeterCompat:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        hist = confusion_matrix(logits, target, self.num_classes, self.ignore_index)
        self.confusion += hist.cpu().numpy()

    def compute(self) -> dict[str, float | list[float | None]]:
        return summarize_metrics(torch.from_numpy(self.confusion))


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    amp: bool,
    scaler: GradScaler | None,
    desc: str,
) -> tuple[dict[str, float | list[float | None]], torch.Tensor]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_samples = 0
    meter = SegmentationMeterCompat(num_classes=num_classes, ignore_index=IGNORE_INDEX)
    hist = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    pbar = tqdm(loader, desc=desc, leave=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bs = images.size(0)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with autocast("cuda", enabled=amp):
                logits = model(images)
                loss = criterion(logits, masks)
            if is_train:
                assert optimizer is not None
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * bs
        n_samples += bs
        batch_hist = confusion_matrix(logits.detach(), masks.detach(), num_classes, IGNORE_INDEX)
        hist += batch_hist.cpu()
        meter.update(logits.detach(), masks.detach())
        metrics = meter.compute()
        pbar.set_postfix(loss=total_loss / max(n_samples, 1), miou=metrics["miou"])

    out = meter.compute()
    out["loss"] = total_loss / max(n_samples, 1)
    return out, hist


def resolve_loss_name(loss_mode: str) -> str:
    return "ce_dice" if loss_mode == "combo" else loss_mode


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, Any], config: dict[str, Any], history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "args": config,
            "history": history,
            "class_names": CLASS_NAMES,
        },
        path,
    )


def plot_training_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    epochs = [record["epoch"] for record in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [record["train_loss"] for record in history], label="train")
    axes[0].plot(epochs, [record["val_loss"] for record in history], label="val")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, [record["train_miou"] for record in history], label="train mIoU")
    axes[1].plot(epochs, [record["val_miou"] for record in history], label="val mIoU")
    axes[1].set_title("mIoU Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(hist: torch.Tensor, class_names: tuple[str, ...], output_path: Path) -> None:
    hist = hist.float()
    row_sum = hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized = (hist / row_sum).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Normalized Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = Path(f"runs/{args.task}_baseline")

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config["num_classes"] = len(CLASS_NAMES)
    config["class_names"] = CLASS_NAMES
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    train_ds = StanfordBackgroundDataset(args.data_root, split="train", image_size=tuple(args.image_size), augment=True)
    val_ds = StanfordBackgroundDataset(args.data_root, split="val", image_size=tuple(args.image_size), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = UNet(in_channels=3, num_classes=len(CLASS_NAMES), base_channels=args.base_channels).to(device)
    criterion = build_loss(resolve_loss_name(args.loss_mode), num_classes=len(CLASS_NAMES), ignore_index=IGNORE_INDEX, ce_weight=args.ce_weight, dice_weight=args.dice_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = GradScaler(enabled=args.amp)

    history: list[dict[str, Any]] = []
    best_miou = -1.0
    best_hist: torch.Tensor | None = None
    best_metrics: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_one_epoch(model, train_loader, criterion, optimizer, device, len(CLASS_NAMES), args.amp, scaler, f"train {epoch}/{args.epochs}")
        val_metrics, val_hist = run_one_epoch(model, val_loader, criterion, None, device, len(CLASS_NAMES), args.amp, None, f"val {epoch}/{args.epochs}")
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_miou": train_metrics["miou"],
            "train_pixel_acc": train_metrics["pixel_acc"],
            "train_mean_acc": train_metrics["mean_acc"],
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_mean_acc": val_metrics["mean_acc"],
            "val_per_class_iou": val_metrics["per_class_iou"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)

        console_row = {
            "epoch": epoch,
            "train_loss": row["train_loss"],
            "train_miou": row["train_miou"],
            "train_pixel_acc": row["train_pixel_acc"],
            "val_loss": row["val_loss"],
            "val_miou": row["val_miou"],
            "val_pixel_acc": row["val_pixel_acc"],
            "val_per_class_iou": row["val_per_class_iou"],
        }
        print(json.dumps(console_row, ensure_ascii=False))

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, row, config, history)
        if val_metrics["miou"] > best_miou:
            best_miou = float(val_metrics["miou"])
            best_metrics = row.copy()
            best_hist = val_hist.clone()
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_metrics, config, history)
            (args.output_dir / "metrics_best.json").write_text(json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_training_curves(history, args.output_dir)
    if best_hist is not None:
        torch.save(best_hist, args.output_dir / "best_confusion_matrix.pt")
        np.save(args.output_dir / "best_confusion_matrix.npy", best_hist.cpu().numpy())
        plot_confusion_matrix(best_hist, tuple(TASK_METADATA[args.task].class_names), args.output_dir / "best_confusion_matrix.png")


if __name__ == "__main__":
    main()
