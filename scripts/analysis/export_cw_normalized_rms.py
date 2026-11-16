#!/usr/bin/env python3
"""Export C&W initial-constant 0.01 ASR and normalized RMS distortion.

The saved C&W L2 distances are measured in model-input [0, 1] space.
For an input with N elements, the per-element RMS distortion is L2 / sqrt(N).
Distortion summaries are calculated over successful attacks only.
"""

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "adv_output" / "cw" / "cw_const_0p01_normalized_rms.csv"

DATASETS = (
    "pneumoniamnist",
    "breastmnist",
    "nodulemnist3d",
    "bloodmnist",
)
MODELS = (
    ("resnet18", "ResNet-18"),
    ("resnet50", "ResNet-50"),
    ("vit_224", "MedViT"),
    ("bpcaps", "BP-CapsNet"),
    ("caps", "DR-CapsNet"),
)

NATIVE_SHAPES = {
    "pneumoniamnist": (1, 28, 28),
    "breastmnist": (1, 28, 28),
    "nodulemnist3d": (28, 28, 28),
    "bloodmnist": (3, 28, 28),
}
MEDVIT_SHAPES = {
    "pneumoniamnist": (1, 224, 224),
    "breastmnist": (1, 224, 224),
    "nodulemnist3d": (28, 224, 224),
    "bloodmnist": (3, 224, 224),
}

FIELDS = (
    "dataset",
    "model",
    "status",
    "initial_const",
    "cohort",
    "test_set_cases",
    "attacked_cases",
    "successful_attacks",
    "asr",
    "input_shape",
    "input_elements",
    "median_l2",
    "q1_l2",
    "q3_l2",
    "median_rms",
    "q1_rms",
    "q3_rms",
)


def result_path(dataset: str, model_file: str) -> Path:
    if dataset == "bloodmnist":
        blood_model = "vit" if model_file == "vit_224" else model_file
        return (
            ROOT
            / "adv_output"
            / "cw"
            / "blood_shared300"
            / f"{dataset}_{blood_model}_cw_const_0p01_metrics.npz"
        )
    return ROOT / "adv_output" / "cw" / f"{dataset}_{model_file}_cw_metrics.npz"


def scalar(result, key):
    return result[key].item()


def make_row(dataset: str, model_file: str, model_label: str) -> dict:
    shape = MEDVIT_SHAPES[dataset] if model_file == "vit_224" else NATIVE_SHAPES[dataset]
    elements = math.prod(shape)
    path = result_path(dataset, model_file)
    common = {
        "dataset": dataset,
        "model": model_label,
        "input_shape": "x".join(map(str, shape)),
        "input_elements": elements,
    }
    if not path.is_file():
        return {
            **common,
            "status": "missing_result",
            "initial_const": 0.01,
        }

    with np.load(path, allow_pickle=False) as result:
        successful_l2 = np.asarray(result["successful_l2"], dtype=np.float64)
        if successful_l2.size == 0:
            median_l2 = q1_l2 = q3_l2 = float("nan")
        else:
            median_l2 = float(np.median(successful_l2))
            q1_l2, q3_l2 = np.quantile(successful_l2, (0.25, 0.75))
        scale = math.sqrt(elements)
        blood = dataset == "bloodmnist"
        return {
            **common,
            "status": "complete",
            "initial_const": (
                float(scalar(result, "initial_const")) if blood else 0.01
            ),
            "cohort": (
                str(scalar(result, "selection")) if blood
                else "model-correct_full-test"
            ),
            "test_set_cases": int(scalar(
                result, "test_set_size" if blood else "total_count"
            )),
            "attacked_cases": int(scalar(
                result, "initially_correct_count"
            )),
            "successful_attacks": int(scalar(result, "success_count")),
            "asr": float(scalar(result, "asr")),
            "median_l2": median_l2,
            "q1_l2": float(q1_l2),
            "q3_l2": float(q3_l2),
            "median_rms": median_l2 / scale,
            "q1_rms": float(q1_l2) / scale,
            "q3_rms": float(q3_l2) / scale,
        }


def main() -> None:
    rows = [
        make_row(dataset, model_file, model_label)
        for dataset in DATASETS
        for model_file, model_label in MODELS
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
