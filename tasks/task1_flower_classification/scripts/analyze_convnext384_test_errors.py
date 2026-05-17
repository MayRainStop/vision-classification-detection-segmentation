from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flower_classification.data import Oxford102FlowersDataset, load_samples
from flower_classification.models import build_model
from flower_classification.train import build_transforms, require_torchvision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ConvNeXt-384 test-set errors for Oxford 102 Flowers.")
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing jpg/, imagelabels.mat, and setid.mat.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the final ConvNeXt checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON/figure outputs.")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-examples", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    torch, nn, _ = require_torchvision()
    from torch.utils.data import DataLoader
    from torchvision import transforms

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, eval_tfms = build_transforms(args.image_size, transforms)
    samples = load_samples(args.data_root, "test")
    dataset = Oxford102FlowersDataset(samples, eval_tfms)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, num_classes=102, pretrained=False)
    checkpoint = load_checkpoint(args.checkpoint, torch, device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    confusion = np.zeros((102, 102), dtype=np.int64)
    predictions: list[dict[str, object]] = []
    total_loss = 0.0
    total = 0
    correct = 0
    sample_offset = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = logits.softmax(dim=1)
            confidence, predicted = probs.max(dim=1)

            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            correct += int((predicted == labels).sum().item())
            total += batch_size

            labels_cpu = labels.cpu().numpy()
            predicted_cpu = predicted.cpu().numpy()
            confidence_cpu = confidence.cpu().numpy()
            for batch_index, (true_label, pred_label, conf) in enumerate(zip(labels_cpu, predicted_cpu, confidence_cpu)):
                sample = samples[sample_offset + batch_index]
                confusion[int(true_label), int(pred_label)] += 1
                predictions.append(
                    {
                        "image": sample.path.name,
                        "path": str(sample.path),
                        "true_class": int(true_label) + 1,
                        "pred_class": int(pred_label) + 1,
                        "correct": int(true_label) == int(pred_label),
                        "confidence": float(conf),
                    }
                )
            sample_offset += batch_size

    test_acc = correct / total
    test_loss = total_loss / total
    top_confusions = write_outputs(args, predictions, confusion, test_loss, test_acc)
    plot_confusion_matrix(confusion, figures_dir / "task1_convnext384_confusion_matrix.png")
    plot_top_confusions(top_confusions, figures_dir / "task1_convnext384_top_confusions.png")
    plot_misclassified_examples(predictions, args.num_examples, figures_dir / "task1_convnext384_misclassified_examples.png")

    print(f"test_loss={test_loss:.6f} test_acc={test_acc:.6f} mistakes={total - correct}/{total}")
    print(f"outputs={args.output_dir}")


def load_checkpoint(path: Path, torch_module, device):
    try:
        return torch_module.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location=device)


def write_outputs(
    args: argparse.Namespace,
    predictions: list[dict[str, object]],
    confusion: np.ndarray,
    test_loss: float,
    test_acc: float,
) -> list[dict[str, object]]:
    predictions_path = args.output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "path", "true_class", "pred_class", "correct", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)

    np.savetxt(args.output_dir / "confusion_matrix_counts.csv", confusion, delimiter=",", fmt="%d")

    true_totals = confusion.sum(axis=1)
    pair_examples: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    pair_counts: Counter[tuple[int, int]] = Counter()
    for row in predictions:
        true_class = int(row["true_class"])
        pred_class = int(row["pred_class"])
        if true_class == pred_class:
            continue
        key = (true_class, pred_class)
        pair_counts[key] += 1
        if len(pair_examples[key]) < 5:
            pair_examples[key].append(row)

    top_confusions: list[dict[str, object]] = []
    for rank, ((true_class, pred_class), count) in enumerate(pair_counts.most_common(30), start=1):
        true_total = int(true_totals[true_class - 1])
        top_confusions.append(
            {
                "rank": rank,
                "true_class": true_class,
                "pred_class": pred_class,
                "count": count,
                "true_class_total": true_total,
                "confusion_rate": count / true_total if true_total else 0.0,
                "example_images": "; ".join(str(example["image"]) for example in pair_examples[(true_class, pred_class)]),
            }
        )

    with (args.output_dir / "top_confused_classes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "true_class", "pred_class", "count", "true_class_total", "confusion_rate", "example_images"],
        )
        writer.writeheader()
        writer.writerows(top_confusions)

    summary = {
        "model": args.model,
        "image_size": args.image_size,
        "checkpoint": str(args.checkpoint),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_size": len(predictions),
        "mistakes": sum(1 for row in predictions if not row["correct"]),
        "correct": sum(1 for row in predictions if row["correct"]),
        "top_confusions": top_confusions[:10],
    }
    (args.output_dir / "classification_error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return top_confusions


def plot_confusion_matrix(confusion: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_totals, out=np.zeros_like(confusion, dtype=float), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(12.5, 10.5), dpi=220)
    im = ax.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    ticks = np.arange(0, 102, 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick + 1) for tick in ticks], fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(tick + 1) for tick in ticks], fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("ConvNeXt-Tiny 384px test confusion matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Row-normalized proportion")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_confusions(top_confusions: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = top_confusions[:15]
    labels = [f"{row['true_class']} -> {row['pred_class']}" for row in rows]
    counts = [int(row["count"]) for row in rows]
    rates = [float(row["confusion_rate"]) for row in rows]
    y_pos = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=220)
    colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(rows)))
    ax.barh(y_pos, counts, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of test images")
    ax.set_title("Top confused class pairs")
    max_count = max(counts) if counts else 1
    for index, (count, rate) in enumerate(zip(counts, rates)):
        ax.text(count + 0.08 * max_count, index, f"{count} ({rate:.1%})", va="center", fontsize=8)
    ax.set_xlim(0, max_count * 1.45)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_examples(predictions: list[dict[str, object]], num_examples: int, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    mistakes = [row for row in predictions if not row["correct"]]
    mistakes.sort(key=lambda row: float(row["confidence"]), reverse=True)
    selected = mistakes[:num_examples]
    if not selected:
        return

    cols = 6
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, rows * 2.35), dpi=220)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat:
        ax.axis("off")

    for ax, row in zip(axes_flat, selected):
        image = Image.open(row["path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(
            f"T:{row['true_class']}  P:{row['pred_class']}\nconf={float(row['confidence']):.2f}",
            fontsize=8,
            pad=3,
        )
        ax.axis("off")

    fig.suptitle("High-confidence misclassified test examples", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
