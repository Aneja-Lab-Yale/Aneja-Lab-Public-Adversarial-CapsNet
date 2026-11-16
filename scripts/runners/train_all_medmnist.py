#!/usr/bin/env python3
"""Train every MedMNIST dataset/model combination with resumable checkpoints."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = (
    "pneumoniamnist",
    "breastmnist",
    "nodulemnist3d",
    "bloodmnist",
)
MODELS = ("resnet18", "resnet50", "vit", "caps", "bpcaps")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--gpu-ids", default="0",
        help="Value passed to experiments_medmnist.py; use -1 for CPU",
    )
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True,
        help="Download missing datasets (default: enabled)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-dir", type=Path, default=ROOT / "logs" / "medmnist_training",
    )
    return parser.parse_args()


def checkpoint_path(dataset, model):
    resize_tag = "_224" if model == "vit" else ""
    return ROOT / "checkpoints" / (
        "{}_{}{}_best_model.pth".format(model, dataset, resize_tag)
    )


def command_for(dataset, model, args):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "experiments" / "experiments_medmnist.py"),
        "--data_flag", dataset,
        "--model_flag", model,
        "--num_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--gpu_ids", args.gpu_ids,
        "--run", "{}_{}".format(dataset, model),
    ]
    if model == "vit":
        command.append("--resize")
    if args.download:
        command.append("--download")
    return command


def format_command(command):
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def run_and_log(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    jobs = [
        (dataset, model)
        for dataset in args.datasets
        for model in args.models
    ]
    completed = []
    skipped = []
    failures = []
    for position, (dataset, model) in enumerate(jobs, start=1):
        checkpoint = checkpoint_path(dataset, model)
        label = "{}/{}".format(dataset, model)
        if checkpoint.is_file() and not args.force:
            print(
                "[{}/{}] Skipping {}: checkpoint exists at {}".format(
                    position, len(jobs), label, checkpoint
                ),
                flush=True,
            )
            skipped.append(label)
            continue

        command = command_for(dataset, model, args)
        log_path = args.log_dir / dataset / "{}.log".format(model)
        print(
            "[{}/{}] Training {}\nCommand: {}\nLog: {}".format(
                position, len(jobs), label, format_command(command), log_path
            ),
            flush=True,
        )
        if args.dry_run:
            continue
        exit_code = run_and_log(command, log_path)
        if exit_code or not checkpoint.is_file():
            reason = "exit {}".format(exit_code) if exit_code else "checkpoint missing"
            failures.append((label, reason))
            print("Failed {}: {}".format(label, reason), flush=True)
            if not args.continue_on_error:
                break
            continue
        completed.append(label)

    if args.dry_run:
        print("Dry run complete; no training was started.")
        return
    print("MedMNIST training summary:")
    print("  completed: {}".format(len(completed)))
    print("  skipped: {}".format(len(skipped)))
    print("  failed: {}".format(len(failures)))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
