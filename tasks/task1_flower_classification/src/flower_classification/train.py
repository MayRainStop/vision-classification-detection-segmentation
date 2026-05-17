from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import Oxford102FlowersDataset, load_samples
from .models import SUPPORTED_MODEL_NAMES, build_model, require_torchvision


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float


@dataclass(frozen=True)
class EarlyStoppingDecision:
    improved: bool
    should_stop: bool


@dataclass
class EarlyStoppingState:
    patience: int | None
    min_delta: float = 0.0
    best_score: float = float("-inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def step(self, score: float, epoch: int) -> EarlyStoppingDecision:
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = self.patience is not None and self.patience >= 0 and self.bad_epochs >= self.patience
        return EarlyStoppingDecision(improved=improved, should_stop=should_stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Oxford 102 Flowers classifiers.")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Folder containing jpg/, imagelabels.mat, setid.mat.")
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=SUPPORTED_MODEL_NAMES,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None, help="Override the run folder name. Defaults to the model name.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Train without ImageNet weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train the final classifier head.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="CrossEntropyLoss label smoothing.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many validation epochs without improvement. Leave empty to disable.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimum val_acc gain counted as improvement.")
    parser.add_argument("--use-wandb", action="store_true", help="Log metrics to Weights & Biases if installed.")
    parser.add_argument("--limit-train", type=int, default=None, help="Debug option: use only the first N train samples.")
    parser.add_argument("--limit-val", type=int, default=None, help="Debug option: use only the first N val samples.")
    parser.add_argument("--limit-test", type=int, default=None, help="Debug option: use only the first N test samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch, nn, _ = require_torchvision()
    from torch.utils.data import DataLoader
    from torchvision import transforms

    set_seed(args.seed, torch)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = args.output_dir / (args.run_name or args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    train_tfms, eval_tfms = build_transforms(args.image_size, transforms)
    train_samples = _maybe_limit(load_samples(args.data_root, "train"), args.limit_train)
    val_samples = _maybe_limit(load_samples(args.data_root, "val"), args.limit_val)
    test_samples = _maybe_limit(load_samples(args.data_root, "test"), args.limit_test)

    train_loader = DataLoader(
        Oxford102FlowersDataset(train_samples, train_tfms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Oxford102FlowersDataset(val_samples, eval_tfms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        Oxford102FlowersDataset(test_samples, eval_tfms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, num_classes=102, pretrained=not args.no_pretrained)
    if args.freeze_backbone:
        freeze_backbone(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    wandb_run = init_wandb(args) if args.use_wandb else None

    early_stopping = EarlyStoppingState(args.early_stopping_patience, args.early_stopping_min_delta)
    history: list[EpochMetrics] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, torch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, torch)
        scheduler.step()

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=optimizer.param_groups[0]["lr"],
        )
        history.append(metrics)
        append_metrics(metrics_path, metrics)
        if wandb_run is not None:
            wandb_run.log(asdict(metrics))

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        decision = early_stopping.step(val_acc, epoch)
        if decision.improved:
            torch.save(
                {"model": model.state_dict(), "args": checkpoint_args(args), "epoch": epoch, "val_acc": val_acc},
                run_dir / "best.pt",
            )
        elif decision.should_stop:
            print(
                f"early_stopping epoch={epoch:03d} best_epoch={early_stopping.best_epoch} "
                f"best_val_acc={early_stopping.best_score:.4f} patience={early_stopping.patience}"
            )
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, torch)
    print(f"best_epoch={checkpoint['epoch']} best_val_acc={checkpoint['val_acc']:.4f}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    write_summary(
        run_dir / "summary.json",
        args=args,
        epochs_completed=history[-1].epoch if history else 0,
        best_epoch=int(checkpoint["epoch"]),
        best_val_acc=float(checkpoint["val_acc"]),
        test_loss=test_loss,
        test_acc=test_acc,
        early_stopped=history[-1].epoch < args.epochs if history else False,
    )
    if wandb_run is not None:
        wandb_run.log({"test_loss": test_loss, "test_acc": test_acc})
        wandb_run.finish()


def build_transforms(image_size: int, transforms_module):
    normalize = transforms_module.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = transforms_module.Compose(
        [
            transforms_module.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms_module.RandomHorizontalFlip(),
            transforms_module.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms_module.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms_module.Compose(
        [
            transforms_module.Resize(int(image_size * 1.14)),
            transforms_module.CenterCrop(image_size),
            transforms_module.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def train_one_epoch(model, loader, criterion, optimizer, device, torch_module) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (torch_module.argmax(logits, dim=1) == labels).sum().item()
        total += batch_size
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, torch_module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch_module.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (torch_module.argmax(logits, dim=1) == labels).sum().item()
            total += batch_size
    return total_loss / total, correct / total


def freeze_backbone(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for module_name in ["fc", "head", "heads"]:
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def append_metrics(path: Path, metrics: EpochMetrics) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics)))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(metrics))


def checkpoint_args(args: argparse.Namespace) -> dict[str, str | int | float | bool | None]:
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def write_summary(
    path: Path,
    args: argparse.Namespace,
    epochs_completed: int,
    best_epoch: int,
    best_val_acc: float,
    test_loss: float,
    test_acc: float,
    early_stopped: bool,
) -> None:
    payload = {
        "args": checkpoint_args(args),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "early_stopped": early_stopped,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def init_wandb(args: argparse.Namespace):
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install wandb or remove --use-wandb.") from exc
    return wandb.init(project="oxford102-flower-classification", name=args.model, config=vars(args))


def _maybe_limit(samples, limit: int | None):
    return samples if limit is None else samples[:limit]


if __name__ == "__main__":
    main()
