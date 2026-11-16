#!/usr/bin/env python3
"""Run AutoAttack at epsilon 0.032 for every medical dataset/model pair."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASETS = (
    "pneumoniamnist",
    "breastmnist",
    "nodulemnist3d",
    "bloodmnist",
)
MODELS = ("resnet18", "resnet50", "vit", "caps", "bpcaps")
DEFAULT_EPSILON = 0.032


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-dir", type=Path, default=ROOT / "logs" / "medical_autoattack",
    )
    return parser.parse_args()


def epsilon_tag(epsilon):
    return "{:g}".format(epsilon)


def checkpoint_path(dataset, model):
    resize_tag = "_224" if model == "vit" else ""
    return ROOT / "checkpoints" / (
        "{}_{}{}_best_model.pth".format(model, dataset, resize_tag)
    )


def output_paths(dataset, model, epsilon):
    resize_tag = "_224" if model == "vit" else ""
    stem = "{}_{}{}_autoattack_eps_{}".format(
        dataset, model, resize_tag, epsilon_tag(epsilon)
    )
    directory = ROOT / "adv_output" / "autoattack"
    return directory / (stem + "_acc.npy"), directory / (stem + "_auc.npy")


def output_complete(dataset, model, epsilon):
    try:
        accuracy_path, auc_path = output_paths(dataset, model, epsilon)
        accuracy = np.load(accuracy_path, allow_pickle=False)
        auc = np.load(auc_path, allow_pickle=False)
        return (
            accuracy.shape == (1,)
            and auc.shape == (1,)
            and np.isfinite(accuracy).all()
            and np.isfinite(auc).all()
        )
    except (OSError, ValueError):
        return False


def command_for(dataset, model, args):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "experiments" / "experiments_medmnist.py"),
        "--data_flag", dataset,
        "--model_flag", model,
        "--num_epochs", "0",
        "--model_path", str(checkpoint_path(dataset, model)),
        "--experiment", "sensitivity",
        "--attack_type", "autoattack",
        "--eps", str(args.epsilon),
        "--batch_size", str(args.batch_size),
        "--gpu_ids", str(args.gpu_id),
        "--run", "{}_{}_autoattack_eps_{}".format(
            dataset, model, epsilon_tag(args.epsilon)
        ),
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
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    jobs = [(dataset, model) for dataset in args.datasets for model in args.models]
    missing = [
        checkpoint_path(dataset, model) for dataset, model in jobs
        if not checkpoint_path(dataset, model).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n  " + "\n  ".join(map(str, missing)))

    completed = []
    skipped = []
    failures = []
    for position, (dataset, model) in enumerate(jobs, start=1):
        label = "{}/{}".format(dataset, model)
        if output_complete(dataset, model, args.epsilon) and not args.force:
            print("[{}/{}] Skipping {}: validated output exists".format(
                position, len(jobs), label
            ), flush=True)
            skipped.append(label)
            continue
        command = command_for(dataset, model, args)
        log_path = (
            args.log_dir / "eps_{}".format(epsilon_tag(args.epsilon))
            / dataset / "{}.log".format(model)
        )
        print("[{}/{}] Running {}\nCommand: {}\nLog: {}".format(
            position, len(jobs), label, format_command(command), log_path
        ), flush=True)
        if args.dry_run:
            continue
        exit_code = run_and_log(command, log_path)
        if exit_code or not output_complete(dataset, model, args.epsilon):
            failures.append(label)
            if not args.continue_on_error:
                break
        else:
            completed.append(label)

    if args.dry_run:
        print("Dry run complete; no AutoAttack jobs were started.")
        return
    print("Medical AutoAttack summary: completed={}, skipped={}, failed={}".format(
        len(completed), len(skipped), len(failures)
    ))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
