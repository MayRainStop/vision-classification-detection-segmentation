from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


CLASS_NAMES = ["sky", "tree", "road", "grass", "water", "building", "mountain", "foreground"]
IGNORE_INDEX = 255


@dataclass(frozen=True)
class DatasetMetadata:
    class_names: tuple[str, ...]
    label_suffix: str

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


TASK_METADATA = {
    "regions": DatasetMetadata(tuple(CLASS_NAMES[:-1] + ["foreground_object"]), "regions"),
}


def read_split_file(data_dir: Path, split: str) -> list[tuple[Path, Path]]:
    split_path = data_dir / "splits" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    pairs: list[tuple[Path, Path]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        img, mask = line.split("\t")[:2]
        pairs.append((Path(img), Path(mask)))
    return pairs


def is_prepared_dataset(root: str | Path) -> bool:
    root = Path(root)
    return (root / "splits").exists() and (root / "processed").exists()


def load_split_ids(image_dir: Path, val_ratio: float = 0.15, seed: int = 42) -> tuple[list[str], list[str]]:
    ids = sorted(path.stem for path in image_dir.glob("*.jpg"))
    if not ids:
        raise FileNotFoundError(f"No JPG images found under {image_dir}")
    rng = np.random.default_rng(seed)
    ids = list(rng.permutation(ids))
    val_size = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:val_size]
    train_ids = ids[val_size:]
    train_ids.sort()
    val_ids.sort()
    return train_ids, val_ids


def load_dataset_records(
    root: str | Path,
    task: str = "regions",
    val_ratio: float = 0.15,
    seed: int = 42,
    split: str | None = None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]] | list[tuple[Path, Path]]:
    root = Path(root)
    metadata = TASK_METADATA[task]
    if is_prepared_dataset(root):
        if split is not None:
            return read_split_file(root, split)
        return read_split_file(root, "train"), read_split_file(root, "val")

    image_dir = root / "images"
    label_dir = root / "labels"
    train_ids, val_ids = load_split_ids(image_dir, val_ratio=val_ratio, seed=seed)

    def pairs(ids: list[str]) -> list[tuple[Path, Path]]:
        return [(image_dir / f"{stem}.jpg", label_dir / f"{stem}.{metadata.label_suffix}.txt") for stem in ids]

    if split == "train":
        return pairs(train_ids)
    if split == "val":
        return pairs(val_ids)
    if split is not None:
        raise ValueError(f"Unsupported split '{split}'")
    return pairs(train_ids), pairs(val_ids)


class StanfordBackgroundDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (320, 240),
        augment: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = tuple(int(x) for x in image_size)
        self.augment = augment
        if is_prepared_dataset(self.data_dir):
            self.pairs = read_split_file(self.data_dir, split)
        else:
            self.pairs = load_dataset_records(self.data_dir, split=split)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() < 0.3:
            factor = float(np.random.uniform(0.8, 1.2))
            image = ImageEnhance.Brightness(image).enhance(factor)
        if np.random.rand() < 0.3:
            factor = float(np.random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        img_path, mask_path = self.pairs[index]
        image = Image.open(img_path).convert("RGB")
        if mask_path.suffix.lower() == ".txt":
            mask_arr = np.loadtxt(mask_path, dtype=np.int64)
            mask_arr[(mask_arr < 0) | (mask_arr >= len(CLASS_NAMES))] = IGNORE_INDEX
            mask = Image.fromarray(mask_arr.astype(np.uint8), mode="L")
        else:
            mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._augment(image, mask)

        width, height = self.image_size
        image = image.resize((width, height), resample=Image.BILINEAR)
        mask = mask.resize((width, height), resample=Image.NEAREST)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        image_tensor = (image_tensor - self.mean) / self.std

        mask_arr = np.asarray(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_arr).long()
        return {"image": image_tensor, "mask": mask_tensor, "image_path": str(img_path), "mask_path": str(mask_path)}


def denormalize(image: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=image.device).view(3, 1, 1)
    return torch.clamp(image * std_t + mean_t, 0.0, 1.0)


class ICCV09SegmentationDataset(StanfordBackgroundDataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path]] | None = None,
        root: str | Path | None = None,
        sample_ids: list[str] | None = None,
        task: str = "regions",
        image_size: tuple[int, int] = (320, 240),
        ignore_index: int = IGNORE_INDEX,
        augment: bool = False,
        transform=None,
    ) -> None:
        del task, ignore_index, transform
        if samples is not None:
            self.data_dir = Path(".")
            self.split = "custom"
            self.image_size = tuple(int(x) for x in image_size)
            self.augment = augment
            self.pairs = samples
            self.mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
            self.std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
            return
        if root is None:
            raise ValueError("Either samples or root must be provided.")
        super().__init__(root, split="train", image_size=image_size, augment=augment)
