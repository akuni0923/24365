import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from wildfire.dataset import load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        config = {"path": ".", "names": ["smoke", "Fire"]}
        for index, split in enumerate(("train", "val", "test")):
            config[split] = f"{split}/images"
            image_dir = self.root / split / "images"
            image_dir.mkdir(parents=True)
            label_dir = image_dir.with_name("labels")
            label_dir.mkdir()
            Image.new("RGB", (32, 32), (index, 0, 0)).save(image_dir / "sample.png")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.3 0.3\n1 0.2 0.2 0.1 0.1\n")
        self.config = self.root / "data.yaml"
        self.config.write_text(yaml.safe_dump(config))

    def test_class_order_and_data_hash_follow_labels(self):
        before = load_dataset(self.config)
        self.assertEqual(before.names, {0: "smoke", 1: "fire"})
        (self.root / "test/labels/sample.txt").write_text("")
        after = load_dataset(self.config)
        self.assertEqual(after.counts["test"]["background_images"], 1)
        self.assertNotEqual(before.manifest()["sha256"], after.manifest()["sha256"])

    def test_renamed_duplicate_across_splits_is_rejected(self):
        image = self.root / "test/images/sample.png"
        image.unlink()
        (image.parent / "renamed.png").write_bytes((self.root / "train/images/sample.png").read_bytes())
        (self.root / "test/labels/sample.txt").rename(self.root / "test/labels/renamed.txt")
        with self.assertRaisesRegex(ValueError, "Identical image across train/test"):
            load_dataset(self.config)

    def test_out_of_range_yolo_coordinate_is_rejected(self):
        (self.root / "val/labels/sample.txt").write_text("0 1.5 0.5 0.3 0.3\n")
        with self.assertRaisesRegex(ValueError, "normalized xywh"):
            load_dataset(self.config)
