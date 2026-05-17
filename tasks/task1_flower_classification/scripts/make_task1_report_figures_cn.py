from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


TASK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = TASK_ROOT / "data"
REPORT_ROOT = TASK_ROOT / "experiment_results" / "final_report"
FIGURES_DIR = REPORT_ROOT / "figures"
METRICS_PATH = (
    TASK_ROOT
    / "experiment_results"
    / "convnext_384_expansion"
    / "convnext_tiny_384_best_seed314_lr0p0001_wd1e-05_bs16_ls0p1"
    / "metrics.csv"
)

BLUE = "#4C78A8"
ORANGE = "#F58518"
TEAL = "#2A9D8F"
RED = "#E15759"
GREEN = "#59A14F"
GRAY = "#6B7280"
LIGHT_GRID = "#E5E7EB"
TEXT = "#222222"


def setup_style() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "Noto Sans SC",
        "SimHei",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D1D5DB",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "axes.linewidth": 0.7,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "savefig.dpi": 240,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(path)


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=LIGHT_GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def image_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def plot_dataset_samples() -> None:
    samples = [
        (1, "06754"),
        (6, "07184"),
        (15, "06370"),
        (23, "03414"),
        (34, "06949"),
        (42, "05714"),
        (51, "01443"),
        (60, "02974"),
        (72, "03593"),
        (84, "02596"),
        (95, "07530"),
        (102, "08024"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(10.4, 7.4), dpi=220)
    for ax, (cls, image_id) in zip(axes.ravel(), samples):
        img_path = DATA_ROOT / "jpg" / f"image_{image_id}.jpg"
        ax.imshow(image_tile(img_path, (360, 280)))
        ax.set_title(f"Class {cls:03d} / Image {image_id}", fontsize=8.5, pad=4)
        ax.axis("off")
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save(fig, "task1_dataset_samples.png")


def plot_resolution_stability() -> None:
    stats = pd.read_csv(REPORT_ROOT / "stability_and_resolution_checks.csv")
    seed_df = pd.read_csv(REPORT_ROOT / "convnext_tiny_384_best10_seed_results.csv")
    comp = stats[stats["group"].isin(["convnext_tiny_224_best10", "convnext_tiny_384_best10"])].copy()
    comp = comp.sort_values("image_size")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.6), dpi=220)

    x = np.arange(len(comp))
    means = comp["test_acc_mean"].to_numpy() * 100
    stds = comp["test_acc_std"].to_numpy() * 100
    ax1.errorbar(x, means, yerr=stds, fmt="o", color=BLUE, ecolor="#A9BEDA", capsize=4, markersize=5)
    ax1.plot(x, means, color=BLUE, linewidth=1.4, alpha=0.75)
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax1.text(i, mean + std + 0.09, f"{mean:.2f}\n±{std:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x, ["224px\n10 runs", "384px\n10 runs"])
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_ylim(93.8, 96.8)
    clean_axes(ax1)

    seed_df = seed_df.sort_values("seed")
    seed_acc = seed_df["test_acc"].to_numpy() * 100
    mean = seed_acc.mean()
    std = seed_acc.std(ddof=1)
    ax2.scatter(seed_df["seed"].astype(str), seed_acc, s=28, color=TEAL, edgecolor="white", linewidth=0.6, zorder=3)
    ax2.axhline(mean, color=RED, linestyle="--", linewidth=1.0, label=f"Mean {mean:.2f}%")
    ax2.axhspan(mean - std, mean + std, color=RED, alpha=0.09, label=f"±1 SD ({std:.2f})")
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_ylim(93.8, 96.8)
    ax2.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax2.legend(loc="lower right", frameon=False)
    clean_axes(ax2)

    fig.tight_layout()
    save(fig, "task1_convnext_resolution_stability.png")


def plot_model_accuracy_comparison() -> None:
    rows = [
        ("ConvNeXt 384px", 95.60, ORANGE),
        ("ConvNeXt 224px", 94.71, ORANGE),
        ("Swin-T", 94.26, TEAL),
        ("ViT-B/16", 92.83, TEAL),
        ("ResNet-34-CBAM", 92.15, BLUE),
        ("ResNet-18 pretrained", 89.97, BLUE),
        ("ResNet-34 scratch", 55.44, "#A0AEC0"),
        ("ResNet-18 scratch", 52.72, "#A0AEC0"),
    ]
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[2] for row in rows]

    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=220)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.30)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for i, value in enumerate(values):
        ax.text(value + 0.28, i, f"{value:.2f}%", va="center", ha="left", fontsize=7.2, color=TEXT)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(50, 98)
    ax.tick_params(axis="y", pad=3)
    clean_axes(ax, "x")
    fig.tight_layout()
    save(fig, "task1_model_accuracy_comparison.png")


def plot_training_curves() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    best_epoch = 31
    early_stop_epoch = int(metrics["epoch"].max())

    fig, ax = plt.subplots(figsize=(8.8, 3.4), dpi=220)
    ax.plot(metrics["epoch"], metrics["train_acc"] * 100, color=BLUE, linewidth=1.7, label="Train")
    ax.plot(metrics["epoch"], metrics["val_acc"] * 100, color=ORANGE, linewidth=1.7, label="Validation")
    ax.axvline(best_epoch, color=GRAY, linestyle="--", linewidth=0.9, alpha=0.75)
    ax.text(best_epoch + 0.6, 64, "Best val.\nepoch 31", color=GRAY, fontsize=7.5, va="bottom")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1, early_stop_epoch)
    ax.set_ylim(0, 104)
    ax.legend(loc="lower right", frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save(fig, "task1_training_accuracy.png")

    fig, ax = plt.subplots(figsize=(8.8, 3.4), dpi=220)
    ax.plot(metrics["epoch"], metrics["train_loss"], color=BLUE, linewidth=1.7, label="Train")
    ax.plot(metrics["epoch"], metrics["val_loss"], color=ORANGE, linewidth=1.7, label="Validation")
    ax.axvline(best_epoch, color=GRAY, linestyle="--", linewidth=0.9, alpha=0.75)
    ax.text(best_epoch + 0.6, metrics["val_loss"].max() * 0.72, "Best val.\nepoch 31", color=GRAY, fontsize=7.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, early_stop_epoch)
    ax.legend(loc="upper right", frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save(fig, "task1_training_loss.png")


def plot_pretraining_ablation() -> None:
    labels = ["ResNet-18", "ResNet-34"]
    scratch = [52.72, 55.44]
    pretrained = [89.97, 89.58]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=220)
    ax.bar(x - width / 2, scratch, width, label="Random init", color="#D66A6A")
    ax.bar(x + width / 2, pretrained, width, label="ImageNet pretrained", color=BLUE)
    for i, (s, p) in enumerate(zip(scratch, pretrained)):
        ax.text(i - width / 2, s + 1.2, f"{s:.2f}", ha="center", va="bottom", fontsize=7.0, color=TEXT)
        ax.text(i + width / 2, p + 1.2, f"{p:.2f}", ha="center", va="bottom", fontsize=7.0, color=TEXT)
        bracket_y = 94.0
        ax.plot(
            [i - width / 2, i - width / 2, i + width / 2, i + width / 2],
            [bracket_y - 1.0, bracket_y, bracket_y, bracket_y - 1.0],
            color=GRAY,
            linewidth=0.7,
        )
        ax.text(i, bracket_y + 1.1, f"+{p - s:.1f} pp", ha="center", va="bottom", fontsize=7.0, color=GRAY)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, handlelength=1.3)
    clean_axes(ax)
    fig.tight_layout()
    save(fig, "task1_pretraining_ablation.png")


def plot_attention_comparison() -> None:
    labels = ["ResNet-18", "ResNet-34"]
    methods = ["Baseline", "SE", "CBAM"]
    values = np.array(
        [
            [89.97, 90.65, 90.86],
            [89.58, 89.72, 92.15],
        ]
    )
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=220)
    for row, label, color, marker in zip(values, labels, [BLUE, ORANGE], ["o", "s"]):
        ax.plot(x, row, color=color, marker=marker, markersize=4.5, linewidth=1.4, label=label)
        ax.text(x[-1] + 0.05, row[-1], f"{row[-1]:.2f}%", va="center", fontsize=7.2, color=color)
    ax.annotate(
        "+0.89 pp",
        xy=(2, values[0, 2]),
        xytext=(1.35, values[0, 2] + 0.45),
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.7},
        fontsize=7.0,
        color=BLUE,
    )
    ax.annotate(
        "+2.57 pp",
        xy=(2, values[1, 2]),
        xytext=(1.35, values[1, 2] + 0.42),
        arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.7},
        fontsize=7.0,
        color=ORANGE,
    )
    ax.set_xticks(x, methods)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(-0.12, 2.55)
    ax.set_ylim(89.2, 92.7)
    ax.legend(frameon=False, loc="upper left")
    clean_axes(ax)
    fig.tight_layout()
    save(fig, "task1_attention_comparison.png")


def pretty_model_name(name: str) -> str:
    mapping = {
        "resnet18": "R18",
        "resnet18_se": "R18-SE",
        "resnet18_cbam": "R18-CBAM",
        "resnet34": "R34",
        "resnet34_se": "R34-SE",
        "resnet34_cbam": "R34-CBAM",
    }
    return mapping.get(name, name)


def plot_hyperparameter_heatmap() -> None:
    df = pd.read_csv(REPORT_ROOT / "all_results_sorted.csv")
    models = ["resnet18", "resnet18_se", "resnet18_cbam", "resnet34", "resnet34_se", "resnet34_cbam"]
    sub = df[
        (df["grid"] == "core_grid")
        & (df["pretrained"] == True)
        & (df["model"].isin(models))
        & (df["learning_rate"].isin([0.001, 0.0003]))
        & (df["batch_size"].isin([32, 64]))
    ].copy()
    best = sub.sort_values("best_val_acc", ascending=False).groupby(["learning_rate", "batch_size"], as_index=False).first()
    lrs = [0.001, 0.0003]
    batches = [32, 64]
    values = np.zeros((len(lrs), len(batches)))
    names = [["" for _ in batches] for _ in lrs]
    for i, lr in enumerate(lrs):
        for j, bs in enumerate(batches):
            row = best[(best["learning_rate"] == lr) & (best["batch_size"] == bs)].iloc[0]
            values[i, j] = row["best_val_acc"] * 100
            names[i][j] = pretty_model_name(str(row["model"]))

    cmap = LinearSegmentedColormap.from_list("task1_blues", ["#F8FBFF", "#A7D8F0", "#4A90C2", "#1F5B93"])
    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=220)
    im = ax.imshow(values, cmap=cmap, vmin=values.min() - 0.2, vmax=values.max() + 0.2)
    ax.set_xticks(range(len(batches)), [str(v) for v in batches])
    ax.set_yticks(range(len(lrs)), [r"$1\times10^{-3}$", r"$3\times10^{-4}$"])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Learning rate")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > values.mean() else TEXT
            ax.text(j, i, f"{values[i, j]:.2f}%\n{names[i][j]}", ha="center", va="center", color=color, fontsize=7.3)
    ax.set_xticks(np.arange(-0.5, len(batches), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(lrs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Best validation accuracy (%)")
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "task1_hyperparameter_heatmap.png")


def plot_top_confusions() -> None:
    df = pd.read_csv(REPORT_ROOT / "convnext384_error_analysis" / "top_confused_classes.csv").head(12)
    labels = [f"True {int(t)} → Pred {int(p)}" for t, p in zip(df["true_class"], df["pred_class"])]
    counts = df["count"].astype(int).to_numpy()
    rates = df["confusion_rate"].to_numpy()
    y = np.arange(len(df))
    colors = plt.cm.Blues(np.linspace(0.82, 0.38, len(df)))

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=220)
    ax.barh(y, counts, color=colors, height=0.34)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassified images")
    max_count = counts.max()
    for i, (count, rate) in enumerate(zip(counts, rates)):
        ax.text(count + 0.14, i, f"{count} imgs, {rate:.1%}", va="center", fontsize=7.1, color=TEXT)
    ax.set_xlim(0, max_count + 3.0)
    clean_axes(ax, "x")
    fig.tight_layout()
    save(fig, "task1_convnext384_top_confusions.png")


def plot_misclassified_examples() -> None:
    df = pd.read_csv(REPORT_ROOT / "convnext384_error_analysis" / "predictions.csv")
    mistakes = df[df["correct"] == False].sort_values("confidence", ascending=False).head(12)

    fig, axes = plt.subplots(3, 4, figsize=(10.4, 7.4), dpi=220)
    for ax, (_, row) in zip(axes.ravel(), mistakes.iterrows()):
        image_path = DATA_ROOT / row["path"]
        ax.imshow(image_tile(image_path, (360, 260)))
        ax.set_title(
            f"T {int(row['true_class'])} / P {int(row['pred_class'])}\nConf. {float(row['confidence']):.2f}",
            fontsize=8.5,
            pad=4,
        )
        ax.axis("off")
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save(fig, "task1_convnext384_misclassified_examples.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_dataset_samples()
    plot_resolution_stability()
    plot_model_accuracy_comparison()
    plot_training_curves()
    plot_pretraining_ablation()
    plot_attention_comparison()
    plot_hyperparameter_heatmap()
    plot_top_confusions()
    plot_misclassified_examples()


if __name__ == "__main__":
    main()
