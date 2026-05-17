from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SoftDiceLoss(nn.Module):
    """Manual multi-class Dice loss for pixel-wise segmentation."""

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=1)
        valid_mask = targets != self.ignore_index

        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
        one_hot = F.one_hot(safe_targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        class_weights: Tensor | None = None,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.dice = SoftDiceLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        total = logits.new_tensor(0.0)

        if self.ce_weight > 0:
            ce = F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                ignore_index=self.ignore_index,
            )
            total = total + self.ce_weight * ce

        if self.dice_weight > 0:
            dice = self.dice(logits, targets)
            total = total + self.dice_weight * dice

        return total
