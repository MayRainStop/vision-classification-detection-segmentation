from __future__ import annotations

import argparse
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15


@dataclass(frozen=True)
class Oxford102Metadata:
    labels: list[int]
    splits: dict[str, list[int]]


@dataclass(frozen=True)
class FlowerSample:
    path: Path
    label: int


def split_name_to_key(split: str) -> str:
    normalized = split.lower().strip()
    aliases = {
        "train": "train",
        "trn": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "tst": "test",
    }
    if normalized not in aliases:
        choices = ", ".join(sorted(aliases))
        raise argparse.ArgumentTypeError(f"Unknown split '{split}'. Choose one of: {choices}")
    return aliases[normalized]


def image_path_for_index(image_dir: Path, matlab_index: int) -> Path:
    return image_dir / f"image_{matlab_index:05d}.jpg"


def build_samples(image_dir: Path, labels: list[int], indices: Iterable[int]) -> list[FlowerSample]:
    samples: list[FlowerSample] = []
    for matlab_index in indices:
        if matlab_index < 1 or matlab_index > len(labels):
            raise ValueError(f"Image index {matlab_index} is outside label range 1..{len(labels)}")

        path = image_path_for_index(image_dir, matlab_index)
        if not path.exists():
            raise FileNotFoundError(f"Missing image file: {path}")

        samples.append(FlowerSample(path=path, label=labels[matlab_index - 1] - 1))
    return samples


def load_oxford102_metadata(data_root: str | Path) -> Oxford102Metadata:
    root = Path(data_root)
    labels_mat = _read_mat_v5(root / "imagelabels.mat")
    setid_mat = _read_mat_v5(root / "setid.mat")

    labels = [int(value) for value in labels_mat["labels"]]
    splits = {
        "train": [int(value) for value in setid_mat["trnid"]],
        "val": [int(value) for value in setid_mat["valid"]],
        "test": [int(value) for value in setid_mat["tstid"]],
    }
    return Oxford102Metadata(labels=labels, splits=splits)


def load_samples(data_root: str | Path, split: str) -> list[FlowerSample]:
    root = Path(data_root)
    metadata = load_oxford102_metadata(root)
    key = split_name_to_key(split)
    return build_samples(root / "jpg", metadata.labels, metadata.splits[key])


class Oxford102FlowersDataset:
    def __init__(self, samples: list[FlowerSample], transform: Callable | None = None) -> None:
        self.samples = samples
        self.transform = transform

    @classmethod
    def from_root(cls, data_root: str | Path, split: str, transform: Callable | None = None):
        return cls(load_samples(data_root, split), transform=transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


def _read_mat_v5(path: Path) -> dict[str, list[int] | list[float]]:
    data = path.read_bytes()
    if len(data) < 128 or not data.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError(f"{path} is not a MATLAB 5.0 MAT-file")

    endian = data[126:128]
    if endian == b"IM":
        order = "<"
    elif endian == b"MI":
        order = ">"
    else:
        raise ValueError(f"Unsupported MATLAB endian marker {endian!r} in {path}")

    variables: dict[str, list[int] | list[float]] = {}
    _read_elements(data[128:], order, variables)
    return variables


def _read_elements(payload: bytes, order: str, variables: dict[str, list[int] | list[float]]) -> None:
    pos = 0
    while pos + 8 <= len(payload):
        dtype, raw, pos = _read_element(payload, pos, order, align=False)
        if dtype == MI_COMPRESSED:
            _read_elements(zlib.decompress(raw), order, variables)
        elif dtype == MI_MATRIX:
            name, values = _parse_matrix(raw, order)
            variables[name] = values
        elif dtype == 0 and not raw:
            break
        else:
            raise ValueError(f"Unsupported top-level MAT element type {dtype}")


def _read_element(buffer: bytes, pos: int, order: str, align: bool = True) -> tuple[int, bytes, int]:
    tag = buffer[pos : pos + 8]
    first, second = struct.unpack(order + "II", tag)

    small_dtype = first & 0xFFFF
    small_size = first >> 16
    if small_dtype in _NUMERIC_DTYPES | {MI_INT8, MI_UINT8} and 0 < small_size <= 4:
        return small_dtype, tag[4 : 4 + small_size], pos + 8

    dtype = first
    size = second
    start = pos + 8
    end = start + size
    next_pos = end + ((8 - (size % 8)) % 8 if align else 0)
    return dtype, buffer[start:end], next_pos


def _parse_matrix(payload: bytes, order: str) -> tuple[str, list[int] | list[float]]:
    pos = 0
    _, _, pos = _read_element(payload, pos, order)
    dims_dtype, dims_raw, pos = _read_element(payload, pos, order)
    if dims_dtype not in {MI_INT32, MI_UINT32}:
        raise ValueError(f"Unsupported MAT dimensions dtype {dims_dtype}")

    _decode_numeric(dims_dtype, dims_raw, order)
    name_dtype, name_raw, pos = _read_element(payload, pos, order)
    if name_dtype not in {MI_INT8, MI_UINT8}:
        raise ValueError(f"Unsupported MAT variable name dtype {name_dtype}")
    name = name_raw.decode("ascii")

    values_dtype, values_raw, _ = _read_element(payload, pos, order)
    values = _decode_numeric(values_dtype, values_raw, order)
    return name, values


_NUMERIC_DTYPES = {
    MI_INT8,
    MI_UINT8,
    MI_INT16,
    MI_UINT16,
    MI_INT32,
    MI_UINT32,
    MI_SINGLE,
    MI_DOUBLE,
    MI_INT64,
    MI_UINT64,
}


def _decode_numeric(dtype: int, raw: bytes, order: str) -> list[int] | list[float]:
    formats = {
        MI_INT8: ("b", 1),
        MI_UINT8: ("B", 1),
        MI_INT16: ("h", 2),
        MI_UINT16: ("H", 2),
        MI_INT32: ("i", 4),
        MI_UINT32: ("I", 4),
        MI_SINGLE: ("f", 4),
        MI_DOUBLE: ("d", 8),
        MI_INT64: ("q", 8),
        MI_UINT64: ("Q", 8),
    }
    if dtype not in formats:
        raise ValueError(f"Unsupported MAT numeric dtype {dtype}")

    fmt, size = formats[dtype]
    if len(raw) % size != 0:
        raise ValueError(f"MAT numeric payload has invalid byte length {len(raw)} for dtype {dtype}")

    count = len(raw) // size
    return list(struct.unpack(order + (fmt * count), raw))
