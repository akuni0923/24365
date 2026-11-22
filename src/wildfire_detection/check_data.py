"""Validate and fingerprint the YOLO detection dataset used by an experiment."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from PIL import Image

SPLITS = ("train", "val", "test")
IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_labels(path, names):
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            class_id, *box = map(float, line.split())
            if class_id not in names or len(box) != 4 or not all(
                math.isfinite(value) and 0 <= value <= 1 for value in box
            ) or min(box[2:]) <= 0:
                raise ValueError
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: expected a valid class ID and normalized xywh") from error
        labels.append((int(class_id), *box))
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: duplicate annotation rows")
    return labels


@dataclass
class ImageRecord:
    image: str
    label: str
    image_sha256: str
    label_sha256: str


@dataclass
class Dataset:
    root: Path
    names: dict[int, str]
    images: dict[str, list[ImageRecord]]
    counts: dict[str, dict]

    def write_yolo_config(self, run_dir):
        config = {"path": str(self.root), "names": self.names}
        for split, records in self.images.items():
            split_file = run_dir / f"{split}.txt"
            split_file.write_text(
                "".join(f"{self.root / record.image}\n" for record in records), encoding="utf-8"
            )
            config[split] = str(split_file)
        (run_dir / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def manifest(self):
        content = {
            "classes": self.names,
            "splits": {
                split: [asdict(record) for record in records]
                for split, records in self.images.items()
            },
        }
        fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        return {"sha256": fingerprint, "counts": self.counts, **content}


def load_dataset(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    root = config.get("path", ".")
    if not isinstance(root, str) or not root:
        raise ValueError("Dataset path must be a directory path.")
    root = (config_path.parent / root).resolve()
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or set(names) != {0, 1} or not all(
        isinstance(name, str) for name in names.values()
    ):
        raise ValueError("Dataset names must map integer IDs 0 and 1.")
    names = {index: name.strip().lower() for index, name in names.items()}
    if set(names.values()) != {"fire", "smoke"} or config.get("nc", 2) != 2:
        raise ValueError("Expected exactly two dataset classes: fire and smoke.")

    images, counts = {}, {}
    # Track identical bytes across splits, including renamed files.
    seen_content = {}
    for split in SPLITS:
        relative = config.get(split)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{split}: expected a directory path ending in /images.")
        image_dir = (root / relative).resolve()
        if not image_dir.is_relative_to(root) or image_dir.name != "images" or not image_dir.is_dir():
            raise ValueError(f"{split}: expected an existing images directory inside the dataset: {image_dir}")
        label_dir = image_dir.with_name("labels")
        records, label_paths = [], set()
        instances = dict.fromkeys(names.values(), 0)
        background_images = 0
        for image in sorted(image_dir.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_FORMATS:
                continue
            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing label for {image}. Verified background images need an empty label file.")
            if label in label_paths:
                raise ValueError(f"Multiple images map to the same label: {label}")
            label_paths.add(label)
            with Image.open(image) as decoded:
                if decoded.width < 10 or decoded.height < 10:
                    raise ValueError(f"Image dimensions must be at least 10 × 10: {image}")
                decoded.verify()
            image_hash = sha256(image)
            previous_split, previous_image = seen_content.setdefault(image_hash, (split, image))
            if previous_split != split:
                raise ValueError(f"Identical image across {previous_split}/{split}: {previous_image} and {image}")
            labels = read_labels(label, names)
            background_images += int(not labels)
            for class_id, *_ in labels:
                instances[names[class_id]] += 1
            records.append(ImageRecord(
                image.relative_to(root).as_posix(),
                label.relative_to(root).as_posix(),
                image_hash,
                sha256(label),
            ))
        if not records:
            raise ValueError(f"{split}: no supported images in {image_dir}")
        orphans = sorted(set(label_dir.rglob("*.txt")) - label_paths)
        if orphans:
            raise ValueError(f"{split}: label without a supported image: {orphans[0]}")
        images[split] = records
        counts[split] = {
            "images": len(records),
            "background_images": background_images,
            "instances": instances,
        }
    if any(count == 0 for count in counts["train"]["instances"].values()):
        raise ValueError("The training split needs at least one instance of each class.")
    return Dataset(root, names, images, counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    args = parser.parse_args()
    try:
        manifest = load_dataset(args.data).manifest()
        summary = {key: manifest[key] for key in ("sha256", "classes", "counts")}
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
