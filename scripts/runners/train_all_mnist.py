#!/usr/bin/env python3
"""Train all MNIST architectures sequentially with resumable checkpoints.

The runner invokes the MNIST experiment program once per model, streams output
to the terminal and a model-specific log, and skips checkpoints that already
exist unless ``--force`` is supplied.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("resnet18", "resnet50", "vit", "caps", "bpcaps")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS),
        help="Models to train, in execution order",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--gpu-ids", default="0",
        help="Value passed to experiments_mnist.py; use -1 for CPU",
    )
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True,
        help="Download MNIST if missing (default: enabled)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain and overwrite models whose checkpoint already exists",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Attempt remaining models if one training command fails",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without training",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=ROOT / "logs" / "mnist_training",
    )
    return parser.parse_args()


def checkpoint_path(model):
    resize_tag = "_224" if model == "vit" else ""
    return ROOT / "checkpoints" / (
        "{}_mnist{}_best_model.pth".format(model, resize_tag)
    )


def command_for(model, args):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "experiments" / "experiments_mnist.py"),
        "--data_flag", "mnist",
        "--model_flag", model,
        "--num_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--gpu_ids", args.gpu_ids,
        "--run", "mnist_{}".format(model),
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
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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

    failures = []
    trained = []
    skipped = []
    for position, model in enumerate(args.models, start=1):
        checkpoint = checkpoint_path(model)
        if checkpoint.is_file() and not args.force:
            print(
                "[{}/{}] Skipping {}: checkpoint exists at {}".format(
                    position, len(args.models), model, checkpoint
                ),
                flush=True,
            )
            skipped.append(model)
            continue

        command = command_for(model, args)
        log_path = args.log_dir / "{}.log".format(model)
        print(
            "[{}/{}] Training {}\nCommand: {}\nLog: {}".format(
                position, len(args.models), model,
                format_command(command), log_path,
            ),
            flush=True,
        )
        if args.dry_run:
            continue
        exit_code = run_and_log(command, log_path)
        if exit_code != 0:
            failures.append((model, exit_code))
            print("{} failed with exit code {}".format(model, exit_code))
            if not args.continue_on_error:
                break
            continue
        if not checkpoint.is_file():
            failures.append((model, "checkpoint not created"))
            print("{} exited successfully but {} is missing".format(
                model, checkpoint
            ))
            if not args.continue_on_error:
                break
            continue
        trained.append(model)
        print("Completed {}: {}".format(model, checkpoint), flush=True)

    if args.dry_run:
        print("Dry run complete; no training was started.")
        return
    print("Training summary:")
    print("  completed: {}".format(", ".join(trained) or "none"))
    print("  skipped: {}".format(", ".join(skipped) or "none"))
    print("  failed: {}".format(
        ", ".join("{} ({})".format(*failure) for failure in failures) or "none"
    ))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
