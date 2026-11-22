"""Capture the inputs and environment associated with each experiment."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, data):
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(f"{encoded}\n", encoding="utf-8")


def git_output(*arguments):
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def add_run_arguments(parser):
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--device", default="cpu", help="cpu, mps, or one visible CUDA index.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--name", help="A unique name for the experiment directory.")


def prepare_run(task, args, dataset, torch, parameters):
    timestamp = datetime.now(timezone.utc)
    name = args.name if args.name is not None else f"{task}-{timestamp:%Y%m%dT%H%M%S%fZ}"
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Run name must be one directory name.")

    device = args.device
    gpu_name = None
    if device.isascii() and device.isdecimal():
        index = int(device)
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable. Check PyTorch and nvidia-smi.")
        gpu_name = torch.cuda.get_device_name(index)
    elif device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this machine.")
    elif device not in {"cpu", "mps"}:
        raise ValueError("Device must be cpu, mps, or a single CUDA index.")

    run_dir = args.runs_dir.resolve() / name
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dataset.manifest()
    dataset.write_yolo_config(run_dir)
    write_json(run_dir / "dataset.json", manifest)

    repo = git_output("rev-parse", "--show-toplevel")
    is_checkout = repo is not None and Path(repo).resolve() == Path(__file__).resolve().parents[2]
    status = git_output("status", "--porcelain") if is_checkout else None
    record = {
        "started_at_utc": timestamp.isoformat(),
        "git_commit": git_output("rev-parse", "HEAD") if is_checkout else None,
        "git_dirty": bool(status) if status is not None else None,
        "command": [sys.executable, "-m", f"wildfire_detection.{task}", *sys.argv[1:]],
        "parameters": parameters,
        "dataset_version": args.dataset_version,
        "dataset_sha256": manifest["sha256"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
    }
    write_json(run_dir / "run.json", record)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run_dir / "environment.txt").write_text(freeze, encoding="utf-8")
    print(f"Run directory: {run_dir}", flush=True)
    return run_dir, record
