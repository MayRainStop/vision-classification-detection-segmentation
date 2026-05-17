import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flower_classification.data import (
    build_samples,
    load_oxford102_metadata,
    split_name_to_key,
)
from flower_classification.train import checkpoint_args


TASK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = TASK_ROOT / "data"


class Oxford102MetadataTests(unittest.TestCase):
    def test_loads_official_split_counts_and_label_range(self):
        metadata = load_oxford102_metadata(DATA_ROOT)

        self.assertEqual(len(metadata.labels), 8189)
        self.assertEqual(len(metadata.splits["train"]), 1020)
        self.assertEqual(len(metadata.splits["val"]), 1020)
        self.assertEqual(len(metadata.splits["test"]), 6149)
        self.assertEqual(min(metadata.labels), 1)
        self.assertEqual(max(metadata.labels), 102)

    def test_build_samples_maps_matlab_indices_to_image_paths_and_zero_based_labels(self):
        with patch.object(Path, "exists", return_value=True):
            samples = build_samples(
                image_dir=DATA_ROOT / "jpg",
                labels=[5, 10, 99],
                indices=[1, 3],
            )

        self.assertEqual(
            [(sample.path.name, sample.label) for sample in samples],
            [("image_00001.jpg", 4), ("image_00003.jpg", 98)],
        )

    def test_split_aliases_are_normalized(self):
        self.assertEqual(split_name_to_key("valid"), "val")
        self.assertEqual(split_name_to_key("validation"), "val")
        self.assertEqual(split_name_to_key("val"), "val")
        self.assertEqual(split_name_to_key("test"), "test")

    def test_checkpoint_args_are_serializable_safe_values(self):
        args = SimpleNamespace(data_root=Path("."), epochs=1, use_wandb=False)

        payload = checkpoint_args(args)

        self.assertEqual(payload["data_root"], ".")
        self.assertEqual(payload["epochs"], 1)
        self.assertIs(payload["use_wandb"], False)


if __name__ == "__main__":
    unittest.main()
