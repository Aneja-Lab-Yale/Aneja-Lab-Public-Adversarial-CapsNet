#!/usr/bin/env python3
"""Generate PGD epsilon=0.016 source sets and evaluate cross-model transfer."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASETS = (
    "mnist",
    "pneumoniamnist",
    "breastmnist",
    "nodulemnist3d",
    "bloodmnist",
)
MODELS = ("resnet18", "resnet50", "vit", "caps", "bpcaps")
DEFAULT_EPSILON = 0.016


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--source-models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=665)
    parser.add_argument("--mnist-subset-size", type=int, default=1000)
    parser.add_argument("--mnist-subset-seed", type=int, default=665)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-dir", type=Path, default=ROOT / "logs" / "pgd_transfer",
    )
    return parser.parse_args()


def epsilon_tag(epsilon):
    return "{:g}".format(epsilon)


def checkpoint_path(dataset, model):
    resize_tag = "_224" if model == "vit" else ""
    return ROOT / "checkpoints" / (
        "{}_{}{}_best_model.pth".format(model, dataset, resize_tag)
    )


def archive_path(dataset, source_model, epsilon):
    return ROOT / "transfer_attack_data" / (
        "{}_{}_pgd_eps_{}_test.npz".format(
            dataset, source_model, epsilon_tag(epsilon)
        )
    )


def result_path(dataset, source_model, epsilon):
    return ROOT / "adv_output" / "transfer" / (
        "{}_{}_pgd_eps_{}_transfer_metrics.npz".format(
            dataset, source_model, epsilon_tag(epsilon)
        )
    )


def target_models(source_model):
    return [model for model in MODELS if model != source_model]


def archive_complete(dataset, source_model, epsilon):
    path = archive_path(dataset, source_model, epsilon)
    try:
        with np.load(path, allow_pickle=False) as archive:
            images = archive["adversarial_images"]
            labels = archive["labels"].reshape(-1)
            return (
                images.ndim == 4
                and images.shape[0] == labels.shape[0]
                and images.shape[0] > 0
                and np.isfinite(images).all()
                and str(archive["data_flag"].item()) == dataset
                and str(archive["source_model"].item()) == source_model
                and str(archive["attack"].item()) == "pgd"
                and np.isclose(float(archive["epsilon"]), epsilon)
                and float(archive["max_linf"]) <= epsilon + 1e-6
            )
    except (KeyError, OSError, ValueError):
        return False


def result_complete(dataset, source_model, epsilon):
    path = result_path(dataset, source_model, epsilon)
    expected_targets = target_models(source_model)
    try:
        with np.load(path, allow_pickle=False) as result:
            return (
                result["models"].astype(str).tolist() == expected_targets
                and result["auc"].shape == (len(expected_targets),)
                and result["accuracy"].shape == (len(expected_targets),)
                and np.isfinite(result["auc"]).all()
                and np.isfinite(result["accuracy"]).all()
                and str(result["data_flag"].item()) == dataset
                and str(result["source_model"].item()) == source_model
                and str(result["attack"].item()) == "pgd"
                and np.isclose(float(result["epsilon"]), epsilon)
            )
    except (KeyError, OSError, ValueError):
        return False


def generate_command(dataset, source_model, args):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "experiments" / "transfer_attack.py"),
        "generate",
        "--data-flag", dataset,
        "--source-model", source_model,
        "--epsilon", str(args.epsilon),
        "--checkpoint", str(checkpoint_path(dataset, source_model)),
        "--output", str(archive_path(dataset, source_model, args.epsilon)),
        "--batch-size", str(args.batch_size),
        "--gpu-id", str(args.gpu_id),
        "--seed", str(args.seed),
    ]
    if dataset == "mnist":
        command.extend([
            "--test-subset-size", str(args.mnist_subset_size),
            "--test-subset-seed", str(args.mnist_subset_seed),
        ])
    if args.download:
        command.append("--download")
    return command


def evaluate_command(dataset, source_model, args):
    return [
        sys.executable,
        str(ROOT / "scripts" / "experiments" / "transfer_attack.py"),
        "evaluate",
        "--input", str(archive_path(dataset, source_model, args.epsilon)),
        "--models", *target_models(source_model),
        "--output", str(result_path(dataset, source_model, args.epsilon)),
        "--batch-size", str(args.batch_size),
        "--gpu-id", str(args.gpu_id),
        "--seed", str(args.seed),
        "--run", "{}_{}_pgd_transfer".format(dataset, source_model),
    ]


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


def execute(command, log_path, args):
    print("Command: {}\nLog: {}".format(format_command(command), log_path), flush=True)
    if args.dry_run:
        return True
    return run_and_log(command, log_path) == 0


def main():
    args = parse_args()
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    if args.batch_size < 1 or args.mnist_subset_size < 1:
        raise ValueError("batch and subset sizes must be positive")
    jobs = [
        (dataset, source_model)
        for dataset in args.datasets
        for source_model in args.source_models
    ]
    required_checkpoints = {
        checkpoint_path(dataset, model)
        for dataset, source_model in jobs
        for model in [source_model, *target_models(source_model)]
    }
    missing = sorted(path for path in required_checkpoints if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n  " + "\n  ".join(map(str, missing)))

    failures = []
    completed = 0
    skipped = 0
    for position, (dataset, source_model) in enumerate(jobs, start=1):
        label = "{}/{}".format(dataset, source_model)
        print("[{}/{}] PGD transfer source {}".format(
            position, len(jobs), label
        ), flush=True)
        archive_ok = archive_complete(dataset, source_model, args.epsilon)
        if not archive_ok or args.force:
            command = generate_command(dataset, source_model, args)
            log_path = args.log_dir / dataset / (source_model + "_generate.log")
            if not execute(command, log_path, args):
                failures.append(label + " generation")
                if not args.continue_on_error:
                    break
                continue
            archive_ok = args.dry_run or archive_complete(
                dataset, source_model, args.epsilon
            )
        else:
            print("Validated source archive exists; skipping generation.")
        if not archive_ok:
            failures.append(label + " invalid archive")
            if not args.continue_on_error:
                break
            continue

        if result_complete(dataset, source_model, args.epsilon) and not args.force:
            print("Validated transfer metrics exist; skipping evaluation.")
            skipped += 1
            continue
        command = evaluate_command(dataset, source_model, args)
        log_path = args.log_dir / dataset / (source_model + "_evaluate.log")
        if not execute(command, log_path, args):
            failures.append(label + " evaluation")
            if not args.continue_on_error:
                break
            continue
        if not args.dry_run and not result_complete(
                dataset, source_model, args.epsilon):
            failures.append(label + " invalid metrics")
            if not args.continue_on_error:
                break
            continue
        completed += 1

    if args.dry_run:
        print("Dry run complete; no transfer attacks were started.")
        return
    print("PGD transfer summary: completed={}, skipped={}, failed={}".format(
        completed, skipped, len(failures)
    ))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
