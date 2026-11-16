#!/usr/bin/env python3
"""Collect inference benchmarks for every dataset/model checkpoint.

Each job invokes the benchmarking worker, writes a JSON result and log, and is
skipped on later runs when its output validates against the requested settings.
The runner also builds one flat CSV containing every timing mode.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = (
    "mnist",
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=665)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "inference_benchmarks",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=ROOT / "logs" / "inference_benchmarks"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checkpoint_path(dataset, model):
    resize_tag = "_224" if model == "vit" else ""
    return ROOT / "checkpoints" / (
        "{}_{}{}_best_model.pth".format(model, dataset, resize_tag)
    )


def result_path(dataset, model, args):
    return args.output_dir / "{}_{}_batch{}.json".format(
        dataset, model, args.batch_size
    )


def read_result(path):
    with path.open() as input_file:
        return json.load(input_file)


def result_complete(dataset, model, args):
    path = result_path(dataset, model, args)
    if not path.is_file():
        return False
    try:
        result = read_result(path)
        expected_modes = (
            {"end_to_end_with_reconstruction", "classification_only"}
            if model in ("caps", "bpcaps")
            else {"classification"}
        )
        measurements = result["measurements"]
        return (
            result["schema_version"] == 1
            and result["dataset"] == dataset
            and result["model"] == model
            and result["batch_size"] == args.batch_size
            and result["warmup_iterations"] == args.warmup_iterations
            and result["iterations"] == args.iterations
            and result["seed"] == args.seed
            and result["device"] == str(args.device)
            and result["input_source"] == "real_test_examples"
            and {item["mode"] for item in measurements} == expected_modes
            and all(item["mean_latency_ms"] > 0 for item in measurements)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def command_for(dataset, model, args):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmarking" / "benchmark_inference.py"),
        "--dataset", dataset,
        "--model", model,
        "--checkpoint", str(checkpoint_path(dataset, model)),
        "--batch-size", str(args.batch_size),
        "--warmup-iterations", str(args.warmup_iterations),
        "--iterations", str(args.iterations),
        "--device", args.device,
        "--seed", str(args.seed),
        "--output", str(result_path(dataset, model, args)),
    ]
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


def write_summary(args):
    rows = []
    for dataset in args.datasets:
        for model in args.models:
            if not result_complete(dataset, model, args):
                continue
            result = read_result(result_path(dataset, model, args))
            for measurement in result["measurements"]:
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "mode": measurement["mode"],
                    "device": result["device"],
                    "device_name": result["device_name"],
                    "batch_size": result["batch_size"],
                    "input_shape": "x".join(map(str, result["input_shape"])),
                    "parameter_count": result["parameter_count"],
                    "warmup_iterations": result["warmup_iterations"],
                    "iterations": result["iterations"],
                    "mean_latency_ms": measurement["mean_latency_ms"],
                    "median_latency_ms": measurement["median_latency_ms"],
                    "p05_latency_ms": measurement["p05_latency_ms"],
                    "p95_latency_ms": measurement["p95_latency_ms"],
                    "throughput_images_per_second": measurement[
                        "throughput_images_per_second"
                    ],
                })
    summary_path = args.output_dir / "summary_batch{}.csv".format(args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "model", "mode", "device", "device_name", "batch_size",
        "input_shape", "parameter_count", "warmup_iterations", "iterations",
        "mean_latency_ms", "median_latency_ms", "p05_latency_ms",
        "p95_latency_ms", "throughput_images_per_second",
    ]
    with summary_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, len(rows)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if args.iterations < 2:
        raise ValueError("--iterations must be at least 2")

    missing = [
        checkpoint_path(dataset, model)
        for dataset in args.datasets
        for model in args.models
        if not checkpoint_path(dataset, model).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoints:\n  " + "\n  ".join(map(str, missing))
        )

    jobs = [
        (dataset, model)
        for dataset in args.datasets
        for model in args.models
    ]
    completed = []
    skipped = []
    failures = []
    for position, (dataset, model) in enumerate(jobs, start=1):
        if result_complete(dataset, model, args) and not args.force:
            print(
                "[{}/{}] Skipping {}/{}: validated result exists".format(
                    position, len(jobs), dataset, model
                ),
                flush=True,
            )
            skipped.append((dataset, model))
            continue

        command = command_for(dataset, model, args)
        log_path = args.log_dir / "{}_{}_batch{}.log".format(
            dataset, model, args.batch_size
        )
        print(
            "[{}/{}] Running {}/{}\nCommand: {}\nLog: {}".format(
                position, len(jobs), dataset, model,
                format_command(command), log_path,
            ),
            flush=True,
        )
        if args.dry_run:
            continue
        exit_code = run_and_log(command, log_path)
        if exit_code or not result_complete(dataset, model, args):
            reason = "exit {}".format(exit_code) if exit_code else "invalid output"
            failures.append((dataset, model, reason))
            print("Failed {}/{}: {}".format(dataset, model, reason))
            if not args.continue_on_error:
                break
            continue
        completed.append((dataset, model))

    if args.dry_run:
        print("Dry run complete; no benchmarks were started.")
        return
    summary_path, summary_rows = write_summary(args)
    print("Inference benchmark summary:")
    print("  completed: {}".format(len(completed)))
    print("  skipped: {}".format(len(skipped)))
    print("  failed: {}".format(len(failures)))
    print("  summary rows: {} ({})".format(summary_rows, summary_path))
    for dataset, model, reason in failures:
        print("    {}/{}: {}".format(dataset, model, reason))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
