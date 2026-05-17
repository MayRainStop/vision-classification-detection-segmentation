from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.segmentation.dataset import ResizeToTensor, TASK_METADATA
from src.segmentation.model import UNet


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
    parser = argparse.ArgumentParser(description="Run inference with a trained ICCV09 segmentation model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="prediction.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_palette(num_classes: int) -> np.ndarray:
    if num_classes <= len(PALETTE):
        return PALETTE[:num_classes]

    rng = np.random.default_rng(42)
    extra = rng.integers(0, 255, size=(num_classes - len(PALETTE), 3), dtype=np.uint8)
    return np.concatenate([PALETTE, extra], axis=0)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    train_args = checkpoint["args"]
    task = train_args["task"]
    num_classes = TASK_METADATA[task].num_classes

    model = UNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=train_args["base_channels"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    transform = ResizeToTensor(tuple(train_args["image_size"]))
    image_tensor, _ = transform(image, np.zeros((image.height, image.width), dtype=np.int64))
    image_tensor = image_tensor.unsqueeze(0).to(args.device)

    with torch.no_grad():
        pred = model(image_tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_image = Image.fromarray(pred, mode="L").resize(original_size, Image.NEAREST)
    pred_array = np.asarray(pred_image, dtype=np.int64)
    palette = build_palette(num_classes)
    color_mask = palette[pred_array]

    output = Image.fromarray(color_mask)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)


if __name__ == "__main__":
    main()
