"""Report standard AP and a separate, fixed-confidence operating point."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from .check_data import load_dataset, read_labels, sha256
from .run_records import add_run_arguments, prepare_run, write_json

AP_CONFIDENCE_MIN = 0.001


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None


def iou(left, right):
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(predictions, ground_truth, match_iou):
    """Match in descending confidence to the best unused box of the same class."""
    unmatched = dict(enumerate(ground_truth))
    tp, fp = [], []
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        candidates = (
            (iou(prediction.xyxy, target.xyxy), index)
            for index, target in unmatched.items()
            if target.class_id == prediction.class_id
        )
        overlap, best = max(candidates, default=(-1.0, -1), key=lambda pair: pair[0])
        if overlap >= match_iou:
            del unmatched[best]
            tp.append(prediction)
        else:
            fp.append(prediction)
    return tp, fp, list(unmatched.values())


def target_boxes(label, names):
    return [
        Box(class_id, (
            max(0.0, x - w / 2), max(0.0, y - h / 2),
            min(1.0, x + w / 2), min(1.0, y + h / 2),
        ))
        for class_id, x, y, w, h in read_labels(label, names)
    ]


def save_example(image, output, names, false_positives, false_negatives):
    with Image.open(image) as original:
        canvas = original.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for kind, color, boxes in (("FP prediction", "#ff4848", false_positives), ("FN ground truth", "#fcd34d", false_negatives)):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            pixels = (x1 * canvas.width, y1 * canvas.height, x2 * canvas.width, y2 * canvas.height)
            draw.rectangle(pixels, outline=color, width=3)
            label = f"{kind} {names[box.class_id]}"
            if box.confidence is not None:
                label += f" {box.confidence:.2f}"
            draw.text((pixels[0], pixels[1]), label, fill=color, stroke_width=1, stroke_fill="black")
    output.parent.mkdir(exist_ok=True)
    canvas.save(output)


def summarize_counts(totals):
    summary = {}
    for name, counts in totals.items():
        tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
        summary[name] = {
            **counts,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    return summary


def evaluate_operating_point(model, dataset, args, run_dir):
    records = {str(dataset.root / record.image): record for record in dataset.images[args.split]}
    source = run_dir / f"{args.split}.txt"
    totals = {name: dict.fromkeys(("tp", "fp", "fn"), 0) for name in dataset.names.values()}
    examples = 0
    predictions = model.predict(
        source=str(source), stream=True, verbose=False, device=args.device, imgsz=args.imgsz,
        batch=args.batch, rect=False, conf=args.confidence, iou=args.nms_iou,
        project=str(run_dir), name="predict",
    )
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as errors:
        for result in predictions:
            image = Path(result.path).absolute()
            record = records.pop(str(image), None)
            if record is None:
                raise ValueError(f"Unexpected or repeated prediction input: {image}")
            boxes = [
                Box(int(class_id), tuple(coords), confidence)
                for class_id, coords, confidence in zip(
                    result.boxes.cls.tolist(), result.boxes.xyxyn.tolist(), result.boxes.conf.tolist(), strict=True
                )
            ]
            truth = target_boxes(dataset.root / record.label, dataset.names)
            tp, fp, fn = match_detections(boxes, truth, args.match_iou)
            for key, matched in (("tp", tp), ("fp", fp), ("fn", fn)):
                for box in matched:
                    totals[dataset.names[box.class_id]][key] += 1
            if fp or fn:
                example = {"image": record.image, "fp": [asdict(box) for box in fp], "fn": [asdict(box) for box in fn]}
                if examples < args.examples:
                    examples += 1
                    output = run_dir / "examples" / f"{examples:03d}.jpg"
                    save_example(image, output, dataset.names, fp, fn)
                    example["render"] = output.relative_to(run_dir).as_posix()
                errors.write(json.dumps(example) + "\n")
    if records:
        raise ValueError(f"No prediction result for {len(records)} images, including {next(iter(records))}")
    return {
        "confidence_min": args.confidence,
        "matching_iou_min": args.match_iou,
        "nms_iou": args.nms_iou,
        "classes": summarize_counts(totals),
    }


def evaluate_ap(model, dataset, args, run_dir):
    metrics = model.val(
        data=str(run_dir / "data.yaml"), split=args.split, device=args.device, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, rect=False, conf=AP_CONFIDENCE_MIN,
        iou=args.nms_iou, plots=True,
        project=str(run_dir), name="ap",
    )
    instances = dataset.counts[args.split]["instances"]
    expected_counts = [instances[dataset.names[index]] for index in range(len(dataset.names))]
    if list(metrics.nt_per_class) != expected_counts:
        raise ValueError("Ultralytics AP label counts differ from the audited split. Check the label cache and dataset.")
    classes = {
        dataset.names[int(class_id)]: {"ap50": float(ap50), "ap50_95": float(ap)}
        for class_id, ap50, ap in zip(metrics.box.ap_class_index, metrics.box.ap50, metrics.box.ap, strict=True)
    }
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "confidence_min": AP_CONFIDENCE_MIN,
        "nms_iou": args.nms_iou,
        "classes": {name: classes.get(name) for name in dataset.names.values()},
    }


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("Expected a value greater than 0 and at most 1.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--confidence", type=probability, required=True)
    parser.add_argument("--match-iou", type=probability, default=0.5)
    parser.add_argument("--nms-iou", type=probability, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    try:
        if min(args.imgsz, args.batch) < 1 or min(args.workers, args.examples) < 0:
            raise ValueError("imgsz and batch must be positive. workers and examples cannot be negative.")
        if args.weights.suffix != ".pt" or not args.weights.is_file():
            raise ValueError(f"Expected a local .pt checkpoint: {args.weights}")
        dataset = load_dataset(args.data)
        counts = dataset.counts[args.split]
        if not any(counts["instances"].values()):
            raise ValueError(f"{args.split}: AP requires at least one annotated object.")

        import torch
        from ultralytics import YOLO

        parameters = {
            "device": args.device,
            "weights": str(args.weights),
            "split": args.split,
            "confidence": args.confidence,
            "match_iou": args.match_iou,
            "nms_iou": args.nms_iou,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "examples": args.examples,
        }
        run_dir, record = prepare_run("evaluate", args, dataset, torch, parameters)
        record["input_checkpoint_sha256"] = sha256(args.weights)
        write_json(run_dir / "run.json", record)
        model = YOLO(str(args.weights.resolve()), task="detect")
        names = {class_id: name.strip().lower() for class_id, name in model.names.items()}
        if names != dataset.names:
            raise ValueError(f"Checkpoint class IDs {names} differ from dataset classes {dataset.names}.")

        report = {
            "split": args.split,
            "counts": counts,
            "ap": evaluate_ap(model, dataset, args, run_dir),
            "operating_point": evaluate_operating_point(model, dataset, args, run_dir),
        }
        write_json(run_dir / "metrics.json", report)
        record["result"] = {"metrics": "metrics.json"}
        write_json(run_dir / "run.json", record)
        print(json.dumps(report, indent=2), flush=True)
    except (ImportError, OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
