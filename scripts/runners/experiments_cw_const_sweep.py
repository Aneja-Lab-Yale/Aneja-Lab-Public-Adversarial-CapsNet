#!/usr/bin/env python3
"""Sweep the CW-L2 trade-off constant on reproducible test subsets."""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import medmnist
import numpy as np
import torch
from art.attacks.evasion import CarliniL2Method
from medmnist import INFO
from torchvision import transforms

from models import bpcapsnet, capsnet
from scripts.experiments.sensitivity import create_attack_classifier, cw_max_iter
from utils import ToTensor3D


DEFAULT_DATASETS = ("breastmnist", "pneumoniamnist", "nodulemnist3d")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--model", choices=("caps", "bpcaps"), default="caps")
    parser.add_argument("--initial-const", nargs="+", type=float,
                        default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--num-cases", type=int, default=32)
    parser.add_argument("--sampling", choices=("balanced", "random"),
                        default="balanced")
    parser.add_argument("--seed", type=int, default=665)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=cw_max_iter)
    parser.add_argument("--binary-search-steps", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("adv_output/cw/const_sweep"))
    parser.add_argument("--verbose", action="store_true",
                        help="Show ART progress bars.")
    return parser.parse_args()


def get_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def get_dataset(data_flag):
    info = INFO[data_flag]
    if "3d" in data_flag:
        transform = transforms.Compose([
            ToTensor3D(),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    data_class = getattr(medmnist, info["python_class"])
    return data_class(
        split="test", transform=transform, download=False, root="data"
    )


def load_model(model_flag, data_flag, channels, classes, device):
    if model_flag == "caps":
        model = capsnet.CapsNet(channels, classes)
    else:
        model = bpcapsnet.BPCapsNet(channels, classes)
    checkpoint = Path("checkpoints") / f"{model_flag}_{data_flag}_best_model.pth"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["net"], strict=True)
    model.to(device).eval()
    return model


def collect_test_set(dataset):
    inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
    targets = np.asarray([
        int(np.asarray(dataset[index][1]).reshape(-1)[0])
        for index in range(len(dataset))
    ])
    return inputs.numpy(), targets


def select_cases(classifier, inputs, targets, count, seed, batch_size,
                 sampling):
    clean_scores = classifier.predict(inputs, batch_size=batch_size)
    clean_predictions = np.argmax(clean_scores, axis=1)
    correct_indices = np.flatnonzero(clean_predictions == targets)
    if count > len(correct_indices):
        raise ValueError(
            f"Requested {count} cases, but only {len(correct_indices)} are correct."
        )
    rng = np.random.default_rng(seed)
    if sampling == "random":
        selected = rng.choice(correct_indices, size=count, replace=False)
    else:
        classes = np.unique(targets)
        allocations = np.full(len(classes), count // len(classes), dtype=int)
        allocations[:count % len(classes)] += 1
        selected_parts = []
        for label, allocation in zip(classes, allocations):
            candidates = correct_indices[targets[correct_indices] == label]
            if allocation > len(candidates):
                raise ValueError(
                    f"Class {label} has only {len(candidates)} correct cases, "
                    f"but balanced sampling requested {allocation}."
                )
            selected_parts.append(
                rng.choice(candidates, size=allocation, replace=False)
            )
        selected = np.concatenate(selected_parts)
    selected = np.sort(selected)
    return selected, clean_scores[selected], clean_predictions[selected]


def const_label(value):
    return f"{value:g}".replace(".", "p").replace("-", "m")


def write_summary(output_dir):
    fields = [
        "dataset", "model", "initial_const", "num_cases", "success_count",
        "asr", "median_l2", "q1_l2", "q3_l2", "mean_l2", "std_l2",
        "max_iter", "binary_search_steps", "confidence", "seed", "sampling",
    ]
    rows = []
    for path in sorted(output_dir.glob("*_metrics.npz")):
        with np.load(path, allow_pickle=False) as result:
            row = {}
            for field in fields:
                value = result[field].item()
                row[field] = value
            rows.append(row)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(args, data_flag, device):
    dataset = get_dataset(data_flag)
    inputs, targets = collect_test_set(dataset)
    classes = len(INFO[data_flag]["label"])
    model = load_model(
        args.model, data_flag, inputs.shape[1], classes, device
    )
    classifier = create_attack_classifier(
        model, inputs.shape[1:], classes, device
    )
    # Offset the seed deterministically so each dataset receives an independent
    # random sample while repeated runs select exactly the same cases.
    dataset_seed = args.seed + sum(map(ord, data_flag))
    selected, clean_scores, clean_predictions = select_cases(
        classifier, inputs, targets, args.num_cases, dataset_seed,
        args.batch_size, args.sampling,
    )
    selected_inputs = inputs[selected]
    selected_targets = targets[selected]

    labels, counts = np.unique(selected_targets, return_counts=True)
    class_counts = dict(zip(labels.tolist(), counts.tolist()))
    print(
        f"{data_flag}: selected test indices {selected.tolist()} "
        f"with class counts {class_counts}",
        flush=True,
    )

    for initial_const in args.initial_const:
        label = const_label(initial_const)
        output_path = (
            args.output_dir
            / f"{data_flag}_{args.model}_cw_const_{label}_metrics.npz"
        )
        if output_path.exists():
            print(f"Skipping existing result: {output_path}", flush=True)
            continue

        print(
            f"Running {data_flag} {args.model}: initial_const={initial_const:g}",
            flush=True,
        )
        attacker = CarliniL2Method(
            classifier=classifier,
            confidence=0.0,
            targeted=False,
            learning_rate=0.01,
            binary_search_steps=args.binary_search_steps,
            max_iter=args.max_iter,
            initial_const=initial_const,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        adversarial_inputs = attacker.generate(x=selected_inputs)
        adversarial_scores = classifier.predict(
            adversarial_inputs, batch_size=args.batch_size
        )
        adversarial_predictions = np.argmax(adversarial_scores, axis=1)
        successful = adversarial_predictions != selected_targets
        l2 = np.linalg.norm(
            ((adversarial_inputs - selected_inputs) / 2.0).reshape(
                args.num_cases, -1
            ),
            axis=1,
        )
        successful_l2 = l2[successful]
        if successful_l2.size:
            q1_l2, median_l2, q3_l2 = np.percentile(
                successful_l2, (25, 50, 75)
            )
            mean_l2 = np.mean(successful_l2)
            std_l2 = np.std(successful_l2)
        else:
            q1_l2 = median_l2 = q3_l2 = mean_l2 = std_l2 = np.nan

        np.savez_compressed(
            output_path,
            dataset=data_flag,
            model=args.model,
            initial_const=initial_const,
            num_cases=args.num_cases,
            success_count=int(successful.sum()),
            asr=float(successful.mean()),
            median_l2=float(median_l2),
            q1_l2=float(q1_l2),
            q3_l2=float(q3_l2),
            mean_l2=float(mean_l2),
            std_l2=float(std_l2),
            max_iter=args.max_iter,
            binary_search_steps=args.binary_search_steps,
            confidence=0.0,
            seed=dataset_seed,
            sampling=args.sampling,
            selected_indices=selected,
            targets=selected_targets,
            clean_scores=clean_scores,
            clean_predictions=clean_predictions,
            adversarial_scores=adversarial_scores,
            adversarial_predictions=adversarial_predictions,
            successful=successful,
            l2=l2,
        )
        write_summary(args.output_dir)
        print(
            f"Saved {output_path}: ASR={successful.mean():.3f} "
            f"({successful.sum()}/{args.num_cases}), median L2={median_l2:.6f}",
            flush=True,
        )


def main():
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError("--num-cases must be positive.")
    if any(value <= 0 for value in args.initial_const):
        raise ValueError("Every --initial-const value must be positive.")
    if args.max_iter <= 0 or args.binary_search_steps <= 0:
        raise ValueError("Iteration counts must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"Using device: {device}", flush=True)
    for data_flag in args.datasets:
        run_dataset(args, data_flag, device)


if __name__ == "__main__":
    main()
