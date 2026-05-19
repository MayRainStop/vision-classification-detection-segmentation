from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def confusion_matrix(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> Tensor:
    preds = torch.argmax(logits, dim=1)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    hist = torch.bincount(targets * num_classes + preds, minlength=num_classes * num_classes)
    return hist.view(num_classes, num_classes)


def summarize_metrics(hist: Tensor) -> dict[str, float | list[float | None]]:
    cm = hist.detach().cpu().numpy().astype(np.float64)
    tp = np.diag(cm)
    pos_gt = cm.sum(axis=1)
    pos_pred = cm.sum(axis=0)
    union = pos_gt + pos_pred - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
    pixel_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0
    mean_acc_per_class = np.divide(tp, pos_gt, out=np.full_like(tp, np.nan), where=pos_gt > 0)
    mean_acc = float(np.nanmean(mean_acc_per_class)) if np.any(~np.isnan(mean_acc_per_class)) else 0.0
    return {
        "miou": miou,
        "pixel_acc": pixel_acc,
        "mean_acc": mean_acc,
        "per_class_iou": [float(x) if not np.isnan(x) else None for x in iou],
    }
