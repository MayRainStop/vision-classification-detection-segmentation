from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.dataset import (
    TASK_METADATA,
    ICCV09SegmentationDataset,
    compute_class_weights,
    load_split_ids,
)
from src.segmentation.losses import SegmentationLoss
from src.segmentation.metrics import confusion_matrix, summarize_metrics
from src.segmentation.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation model on ICCV09Data.")
    parser.add_argument("--data-root", type=str, default="iccv09Data")
    parser.add_argument("--task", type=str, default="regions", choices=sorted(TASK_METADATA))
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("W", "H"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--loss-mode", type=str, default="combo", choices=("ce", "dice", "combo"))
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, list[str]]:
    data_root = Path(args.data_root)
    train_ids, val_ids = load_split_ids(data_root / "images", args.val_ratio, args.seed)

    train_set = ICCV09SegmentationDataset(
        root=data_root,
        sample_ids=train_ids,
        task=args.task,
        image_size=tuple(args.image_size),
        ignore_index=args.ignore_index,
    )
    val_set = ICCV09SegmentationDataset(
        root=data_root,
        sample_ids=val_ids,
        task=args.task,
        image_size=tuple(args.image_size),
        ignore_index=args.ignore_index,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ids


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    num_classes: int,
    ignore_index: int,
) -> tuple[dict[str, float], torch.Tensor]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    hist = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    progress = tqdm(loader, leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        hist += confusion_matrix(logits.detach().cpu(), masks.detach().cpu(), num_classes, ignore_index)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = summarize_metrics(hist)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics, hist


def resolve_loss_weights(args: argparse.Namespace) -> tuple[float, float]:
    if args.loss_mode == "ce":
        return 1.0, 0.0
    if args.loss_mode == "dice":
        return 0.0, 1.0
    return args.ce_weight, args.dice_weight


def plot_training_curves(history: list[dict[str, float]], output_dir: Path) -> None:
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

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.output_dir is None:
        args.output_dir = f"runs/{args.task}_baseline"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = TASK_METADATA[args.task]
    device = torch.device(args.device)
    train_loader, val_loader, train_ids = build_dataloaders(args)

    model = UNet(
        in_channels=3,
        num_classes=metadata.num_classes,
        base_channels=args.base_channels,
    ).to(device)

    class_weights = None
    if not args.disable_class_weights:
        class_weights = compute_class_weights(args.data_root, train_ids, args.task, args.ignore_index).to(device)

    ce_weight, dice_weight = resolve_loss_weights(args)
    criterion = SegmentationLoss(
        class_weights=class_weights,
        ignore_index=args.ignore_index,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_miou = -1.0
    history: list[dict[str, float]] = []
    best_hist: torch.Tensor | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            metadata.num_classes,
            args.ignore_index,
        )
        val_metrics, val_hist = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            num_classes=metadata.num_classes,
            ignore_index=args.ignore_index,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_miou": train_metrics["miou"],
            "train_pixel_acc": train_metrics["pixel_acc"],
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_per_class_iou": val_metrics["per_class_iou"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        latest_path = output_dir / "last.pt"
        torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, latest_path)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            best_hist = val_hist.clone()
            best_path = output_dir / "best.pt"
            torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, best_path)

    with (output_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, ensure_ascii=False, indent=2)

    plot_training_curves(history, output_dir)

    if best_hist is not None:
        torch.save(best_hist, output_dir / "best_confusion_matrix.pt")
        np.save(output_dir / "best_confusion_matrix.npy", best_hist.cpu().numpy())
        plot_confusion_matrix(best_hist, metadata.class_names, output_dir / "best_confusion_matrix.png")


if __name__ == "__main__":
    main()
