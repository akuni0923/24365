"""Run the dataset audit, training, and evaluation with one CLI."""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .dataset import load_dataset, sha256
from .evaluation import evaluate_ap, evaluate_operating_point

ROOT = Path(__file__).resolve().parent.parent


def git_output(*arguments):
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_run(args, dataset, torch):
    timestamp = datetime.now(timezone.utc)
    name = args.name or f"{args.command}-{timestamp:%Y%m%dT%H%M%S%fZ}"
    if name in {".", ".."} or Path(name).name != name:
        raise ValueError("Run name must be one directory name.")
    run_dir = (Path(args.runs_dir).resolve() / name)
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset.write_yolo_config(run_dir)
    manifest = dataset.manifest()
    dump_json(run_dir / "dataset.json", manifest)
    is_repo = git_output("rev-parse", "--show-toplevel") == str(ROOT)
    device = args.device
    gpu_name = None
    if device.isdigit():
        index = int(device)
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable. Check PyTorch and nvidia-smi.")
        gpu_name = torch.cuda.get_device_name(index)
    metadata = {
        "started_at_utc": timestamp.isoformat(),
        "git_commit": git_output("rev-parse", "HEAD") if is_repo else None,
        "git_dirty": bool(git_output("status", "--porcelain")) if is_repo else None,
        "command": [sys.executable, "-m", "wildfire", *sys.argv[1:]],
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset_sha256": manifest["sha256"],
        "dataset_version": args.dataset_version,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
    }
    dump_json(run_dir / "run.json", metadata)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run_dir / "environment.txt").write_text(freeze, encoding="utf-8")
    print(f"Run directory: {run_dir}", flush=True)
    return run_dir, metadata


def train(model, args, run_dir):
    model.train(
        data=str(run_dir / "data.yaml"), device=args.device, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, epochs=args.epochs, seed=args.seed,
        project=str(run_dir), name="train", deterministic=True,
    )
    checkpoint = run_dir / "train" / "weights" / "best.pt"
    return {"checkpoint": str(checkpoint.relative_to(run_dir)), "sha256": sha256(checkpoint)}


def evaluate(model, dataset, args, run_dir):
    names = {class_id: name.strip().lower() for class_id, name in model.names.items()}
    if names != dataset.names:
        raise ValueError(f"Checkpoint class IDs {names} differ from dataset classes {dataset.names}.")
    report = {
        "split": args.split,
        "counts": dataset.counts[args.split],
        "ap": evaluate_ap(model, dataset, args, run_dir),
        "operating_point": evaluate_operating_point(model, dataset, args, run_dir),
    }
    dump_json(run_dir / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return {"metrics": "metrics.json"}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer.")
    return number


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("Expected a value greater than 0 and at most 1.")
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-data", help="Check labels and exact cross-split image duplicates.")
    check.add_argument("--data", default="data.yaml")
    training = commands.add_parser("train", help="Train a YOLO fire/smoke detector.")
    validation = commands.add_parser("evaluate", help="Report AP, fixed-confidence metrics and error examples.")
    for command in (training, validation):
        command.add_argument("--data", default="data.yaml")
        command.add_argument("--dataset-version", required=True)
        command.add_argument("--device", default="cpu", help="cpu, mps, or a single visible CUDA index, e.g. 0.")
        command.add_argument("--imgsz", type=positive_int, default=640)
        command.add_argument("--batch", type=positive_int, default=8)
        command.add_argument("--workers", type=nonnegative_int, default=0)
        command.add_argument("--runs-dir", default="runs")
        command.add_argument("--name")
    training.add_argument("--model", default="yolo11s.pt")
    training.add_argument("--epochs", type=positive_int, default=50)
    training.add_argument("--seed", type=nonnegative_int, default=0)
    validation.add_argument("--weights", type=Path, required=True)
    validation.add_argument("--split", choices=("val", "test"), default="val")
    validation.add_argument("--confidence", type=probability, required=True)
    validation.add_argument("--match-iou", type=probability, default=0.5)
    validation.add_argument("--nms-iou", type=probability, default=0.7)
    validation.add_argument("--examples", type=nonnegative_int, default=12)
    return root


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        dataset = load_dataset(args.data)
        if args.command == "check-data":
            manifest = dataset.manifest()
            print(json.dumps({key: manifest[key] for key in ("sha256", "classes", "counts")}, indent=2))
            return
        if args.device not in {"cpu", "mps"} and not (args.device.isascii() and args.device.isdecimal()):
            raise ValueError("Device must be cpu, mps, or a single CUDA index.")

        import torch
        from ultralytics import YOLO

        if args.command == "evaluate" and not args.weights.is_file():
            raise ValueError(f"Local checkpoint not found: {args.weights}")
        run_dir, metadata = prepare_run(args, dataset, torch)
        source = args.model if args.command == "train" else str(args.weights.resolve())
        model = YOLO(source, task="detect")
        original_checkpoint = Path(getattr(model, "ckpt_path", None) or source)
        metadata["input_checkpoint_sha256"] = sha256(original_checkpoint) if original_checkpoint.is_file() else None
        dump_json(run_dir / "run.json", metadata)
        result = train(model, args, run_dir) if args.command == "train" else evaluate(model, dataset, args, run_dir)
        dump_json(run_dir / "run.json", {**metadata, "result": result})
    except (ImportError, OSError, ValueError, yaml.YAMLError) as error:
        cli.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
