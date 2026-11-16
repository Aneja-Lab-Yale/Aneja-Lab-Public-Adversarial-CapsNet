#!/usr/bin/env python3
"""Calculate clean/adversarial Grad-CAM top-pixel IoU summaries.

For each map, pixels at or above its 80th percentile form the top-20% mask.
Similarity is the intersection over union of the clean and adversarial masks,
matching ``grad_cam.gradcam_similarity(..., percent=20)`` in this repository.
"""

import argparse
import csv
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
MODEL_DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vit": "MedViT",
    "caps": "CapsNet",
    "bpcaps": "BP-CapsNet",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.016)
    parser.add_argument("--top-percent", type=float, default=20.0)
    parser.add_argument(
        "--gradcam-root", type=Path,
        default=ROOT / "gradcam_data" / "pgd_eps_0.016",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output path without an extension; default: "
            "<gradcam-root>/mean_similarity_top20"
        ),
    )
    return parser.parse_args()


def epsilon_tag(epsilon):
    return "{:g}".format(epsilon)


def paired_iou(clean, adversarial, top_percent):
    """Vectorized equivalent of grad_cam.gradcam_similarity."""
    if clean.shape != adversarial.shape or clean.ndim != 3:
        raise ValueError(
            "Expected paired NHW CAM arrays; got {} and {}".format(
                clean.shape, adversarial.shape
            )
        )
    percentile = 100.0 - top_percent
    clean_threshold = np.percentile(
        clean, percentile, axis=(1, 2), keepdims=True
    )
    adversarial_threshold = np.percentile(
        adversarial, percentile, axis=(1, 2), keepdims=True
    )
    clean_mask = clean >= clean_threshold
    adversarial_mask = adversarial >= adversarial_threshold
    intersection = np.logical_and(clean_mask, adversarial_mask).sum(axis=(1, 2))
    union = np.logical_or(clean_mask, adversarial_mask).sum(axis=(1, 2))
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union != 0,
    )


def main():
    args = parse_args()
    if not 0.0 < args.top_percent < 100.0:
        raise ValueError("--top-percent must be in (0, 100)")
    default_root = ROOT / "gradcam_data" / "pgd_eps_0.016"
    if args.gradcam_root == default_root:
        args.gradcam_root = ROOT / "gradcam_data" / (
            "pgd_eps_{}".format(epsilon_tag(args.epsilon))
        )
    if args.output_prefix is None:
        percent_tag = "{:g}".format(args.top_percent)
        args.output_prefix = args.gradcam_root / (
            "mean_similarity_top{}".format(percent_tag)
        )

    rows = []
    mean_matrix = np.empty((len(DATASETS), len(MODELS)), dtype=np.float64)
    std_matrix = np.empty_like(mean_matrix)
    detailed = {}
    for dataset_index, dataset in enumerate(DATASETS):
        for model_index, model in enumerate(MODELS):
            path = args.gradcam_root / dataset / model / "gradcam_maps.npz"
            if not path.is_file():
                raise FileNotFoundError("Missing Grad-CAM data: {}".format(path))
            with np.load(path, allow_pickle=False) as saved:
                clean = saved["clean_gradcam"]
                adversarial = saved["adversarial_gradcam"]
                indices = saved["test_indices"].copy()
                epsilon = float(saved["epsilon"].item())
                target_mode = str(saved["target_mode"].item())
            if not np.isclose(epsilon, args.epsilon):
                raise ValueError("Epsilon mismatch in {}".format(path))

            similarities = paired_iou(clean, adversarial, args.top_percent)
            mean = float(np.mean(similarities))
            std = float(np.std(similarities))
            median = float(np.median(similarities))
            clean_flat = int((np.ptp(clean, axis=(1, 2)) <= 1e-6).sum())
            adversarial_flat = int(
                (np.ptp(adversarial, axis=(1, 2)) <= 1e-6).sum()
            )
            both_flat = int((
                (np.ptp(clean, axis=(1, 2)) <= 1e-6)
                & (np.ptp(adversarial, axis=(1, 2)) <= 1e-6)
            ).sum())
            mean_matrix[dataset_index, model_index] = mean
            std_matrix[dataset_index, model_index] = std
            key = "{}__{}".format(dataset, model)
            detailed[key + "__iou"] = similarities
            detailed[key + "__test_indices"] = indices
            rows.append({
                "dataset": dataset,
                "model": model,
                "model_display_name": MODEL_DISPLAY_NAMES[model],
                "mean_similarity_iou": mean,
                "std_similarity_iou": std,
                "median_similarity_iou": median,
                "num_pairs": len(similarities),
                "top_percent": args.top_percent,
                "epsilon": args.epsilon,
                "target_mode": target_mode,
                "clean_flat_maps": clean_flat,
                "adversarial_flat_maps": adversarial_flat,
                "both_flat_maps": both_flat,
                "source": str(path.resolve()),
            })
            print(
                "{:<18} {:<10} mean IoU={:.6f} (n={})".format(
                    dataset, model, mean, len(similarities)
                ),
                flush=True,
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.output_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        datasets=np.asarray(DATASETS),
        models=np.asarray(MODELS),
        model_display_names=np.asarray([MODEL_DISPLAY_NAMES[m] for m in MODELS]),
        mean_similarity_iou=mean_matrix,
        std_similarity_iou=std_matrix,
        top_percent=np.array(args.top_percent, dtype=np.float64),
        percentile_threshold=np.array(
            100.0 - args.top_percent, dtype=np.float64
        ),
        epsilon=np.array(args.epsilon, dtype=np.float64),
        similarity_metric=np.array("intersection_over_union"),
        mask_rule=np.array("cam >= per-map percentile threshold"),
        **detailed,
    )
    print("Saved {}".format(csv_path))
    print("Saved {}".format(npz_path))


if __name__ == "__main__":
    main()
