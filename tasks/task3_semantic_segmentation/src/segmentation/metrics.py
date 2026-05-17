from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def confusion_matrix(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> Tensor:
    preds = logits.argmax(dim=1)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    indices = targets * num_classes + preds
    hist = torch.bincount(indices, minlength=num_classes * num_classes)
    return hist.view(num_classes, num_classes)


def summarize_metrics(hist: Tensor) -> dict[str, float]:
    hist = hist.float()
    correct = hist.diag().sum()
    total = hist.sum().clamp_min(1.0)
    pixel_acc = (correct / total).item()

    denom = hist.sum(dim=1) + hist.sum(dim=0) - hist.diag()
    iou = hist.diag() / denom.clamp_min(1.0)
    mean_iou = iou.mean().item()

    return {
        "pixel_acc": pixel_acc,
        "miou": mean_iou,
        "per_class_iou": iou.tolist(),
    }
