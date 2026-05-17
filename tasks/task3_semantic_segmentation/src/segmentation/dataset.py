from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


SEMANTIC_CLASSES = (
    "sky",
    "tree",
    "road",
    "grass",
    "water",
    "building",
    "mountain",
    "foreground_object",
)

GEOMETRIC_CLASSES = ("sky", "horizontal", "vertical")


@dataclass(frozen=True)
class DatasetMetadata:
    class_names: tuple[str, ...]
    label_suffix: str

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


TASK_METADATA = {
    "regions": DatasetMetadata(SEMANTIC_CLASSES, "regions"),
    "surfaces": DatasetMetadata(GEOMETRIC_CLASSES, "surfaces"),
}


def load_split_ids(
    image_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    ids = sorted(path.stem for path in image_dir.glob("*.jpg"))
    if not ids:
        raise FileNotFoundError(f"No JPG images found under {image_dir}")

    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_ids = sorted(shuffled[:val_size])
    train_ids = sorted(shuffled[val_size:])
    return train_ids, val_ids


class ResizeToTensor:
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image, mask: np.ndarray) -> tuple[Tensor, Tensor]:
        width, height = self.image_size
        image = image.resize((width, height), Image.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        mask_image = Image.fromarray(mask.astype(np.int64), mode="I")
        mask_image = mask_image.resize((width, height), Image.NEAREST)
        mask_array = np.asarray(mask_image, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor


class ICCV09SegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        root: str | Path,
        sample_ids: list[str],
        task: str = "regions",
        image_size: tuple[int, int] = (256, 256),
        ignore_index: int = 255,
        transform: Callable[[Image.Image, np.ndarray], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        if task not in TASK_METADATA:
            raise ValueError(f"Unsupported task '{task}'. Choose from {sorted(TASK_METADATA)}")

        self.root = Path(root)
        self.sample_ids = sample_ids
        self.task = task
        self.metadata = TASK_METADATA[task]
        self.ignore_index = ignore_index
        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.transform = transform or ResizeToTensor(image_size)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample_id = self.sample_ids[index]
        image_path = self.image_dir / f"{sample_id}.jpg"
        label_path = self.label_dir / f"{sample_id}.{self.metadata.label_suffix}.txt"

        image = Image.open(image_path).convert("RGB")
        mask = np.loadtxt(label_path, dtype=np.int64)
        mask[mask < 0] = self.ignore_index
        image_tensor, mask_tensor = self.transform(image, mask)
        return image_tensor, mask_tensor


def compute_class_weights(
    root: str | Path,
    sample_ids: list[str],
    task: str,
    ignore_index: int = 255,
) -> Tensor:
    if task not in TASK_METADATA:
        raise ValueError(f"Unsupported task '{task}'. Choose from {sorted(TASK_METADATA)}")

    metadata = TASK_METADATA[task]
    label_dir = Path(root) / "labels"
    counts = np.zeros(metadata.num_classes, dtype=np.float64)

    for sample_id in sample_ids:
        mask = np.loadtxt(label_dir / f"{sample_id}.{metadata.label_suffix}.txt", dtype=np.int64)
        mask[mask < 0] = ignore_index
        valid = mask != ignore_index
        valid_mask = mask[valid]
        if valid_mask.size == 0:
            continue
        bincount = np.bincount(valid_mask, minlength=metadata.num_classes)
        counts += bincount[: metadata.num_classes]

    counts = np.maximum(counts, 1.0)
    inv_freq = counts.sum() / counts
    normalized = inv_freq / inv_freq.mean()
    return torch.tensor(normalized, dtype=torch.float32)
