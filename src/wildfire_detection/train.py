"""Train a YOLO fire and smoke detector from a versioned experiment config."""

import argparse
from pathlib import Path

import yaml

from .check_data import load_dataset, sha256
from .run_records import add_run_arguments, prepare_run, write_json


def load_training_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = {"model", "imgsz", "batch", "epochs", "workers", "seed", "deterministic"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError(f"Training YAML must contain exactly: {', '.join(sorted(keys))}.")
    if not isinstance(config["model"], str) or not config["model"].strip():
        raise ValueError("model must be a YOLO checkpoint or architecture path.")
    for key in ("imgsz", "batch", "epochs", "workers", "seed"):
        minimum = 0 if key in {"workers", "seed"} else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer greater than or equal to {minimum}.")
    if type(config["deterministic"]) is not bool:
        raise ValueError("deterministic must be true or false.")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--config", type=Path, default=Path("configs/yolo11s.yaml"))
    parser.add_argument("--model", help="Override the initialization checkpoint or architecture.")
    parser.add_argument("--epochs", type=int, help="Override the config epoch budget.")
    args = parser.parse_args()

    try:
        config = load_training_config(args.config)
        if args.epochs is not None:
            if args.epochs < 1:
                raise ValueError("epochs must be a positive integer.")
            config["epochs"] = args.epochs
        if args.model is not None:
            config["model"] = args.model
        dataset = load_dataset(args.data)

        import torch
        from ultralytics import YOLO

        parameters = {**config, "device": args.device, "config": str(args.config)}
        run_dir, record = prepare_run("train", args, dataset, torch, parameters)
        model = YOLO(config.pop("model"), task="detect")
        initial_weights = getattr(model, "ckpt_path", None)
        if initial_weights:
            record["input_checkpoint_sha256"] = sha256(Path(initial_weights))
            write_json(run_dir / "run.json", record)

        model.train(
            data=str(run_dir / "data.yaml"), device=args.device,
            project=str(run_dir), name="train", **config,
        )
        checkpoint = run_dir / "train" / "weights" / "best.pt"
        record["result"] = {
            "checkpoint": checkpoint.relative_to(run_dir).as_posix(),
            "sha256": sha256(checkpoint),
        }
        write_json(run_dir / "run.json", record)
    except (ImportError, OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
