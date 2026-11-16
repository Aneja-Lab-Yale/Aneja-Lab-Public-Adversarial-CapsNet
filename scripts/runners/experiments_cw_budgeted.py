#!/usr/bin/env python3
"""Run reproducible, budgeted CW-L2 evaluations with per-example outputs.

Three selection modes are supported:

* ``model-correct`` attacks every test example that the current model predicts
  correctly (or a reproducible balanced subset when ``--num-cases`` is set).
* ``shared-correct`` first finds test examples predicted correctly by every
  requested model, creates one class-balanced cohort manifest, and attacks the
  identical test indices for each model.
* ``fixed-cohort`` reuses the indices in an existing cohort manifest and
  attacks the subset that the requested model classifies correctly.

Inputs are normalized to [-1, 1]. Reported L2 distances are converted back to
the model-input [0, 1] pixel space (28x28 for native models and 224x224 for the
current ViT preprocessing).
"""

import argparse
import csv
import gc
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
from torch.utils.data import DataLoader
from torchvision import transforms

from models import bpcapsnet, capsnet
from models.MedViT import MedViT_small
from models.resnets import ResNet18, ResNet50
from scripts.experiments.sensitivity import (
    ClassificationOnlyModel,
    create_attack_classifier,
)
from utils import ToTensor3D


MODEL_CHOICES = ("resnet18", "resnet50", "caps", "bpcaps", "vit")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES,
                        required=True)
    parser.add_argument("--selection", choices=(
        "model-correct", "shared-correct", "fixed-cohort",
    ),
                        default="model-correct")
    parser.add_argument(
        "--cohort-manifest", type=Path,
        help=(
            "Manifest whose selected_indices define a fixed source cohort. "
            "With --selection fixed-cohort, cases the attacked model does not "
            "classify correctly on clean input are excluded before attacking."
        ),
    )
    parser.add_argument(
        "--shared-manifest", type=Path,
        help=(
            "Existing shared-correct cohort manifest to reuse. The manifest "
            "may contain a superset of --models."
        ),
    )
    parser.add_argument(
        "--num-cases", type=int, default=0,
        help="Number of cases to sample; 0 attacks all eligible cases.",
    )
    parser.add_argument("--initial-const", type=float, required=True)
    parser.add_argument("--seed", type=int, default=665)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--binary-search-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"),
                        default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def get_transform(data_flag, model_flag):
    resize = model_flag == "vit"
    if "3d" in data_flag:
        operations = [ToTensor3D()]
        if resize:
            operations.append(
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
            )
        operations.extend([
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return transforms.Compose(operations)

    operations = []
    if resize:
        operations.append(
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
        )
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transforms.Compose(operations)


def get_dataset(data_flag, model_flag, download):
    data_class = getattr(medmnist, INFO[data_flag]["python_class"])
    return data_class(
        split="test",
        transform=get_transform(data_flag, model_flag),
        download=download,
        root="data",
    )


def dataset_targets(dataset):
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels).reshape(-1).astype(np.int64)
    return np.asarray([
        int(np.asarray(dataset[index][1]).reshape(-1)[0])
        for index in range(len(dataset))
    ], dtype=np.int64)


def checkpoint_path(model_flag, data_flag):
    suffix = "_224" if model_flag == "vit" else ""
    return Path("checkpoints") / f"{model_flag}_{data_flag}{suffix}_best_model.pth"


def load_model(model_flag, data_flag, channels, classes, device):
    if model_flag == "resnet18":
        model = ResNet18(in_channels=channels, num_classes=classes)
    elif model_flag == "resnet50":
        model = ResNet50(in_channels=channels, num_classes=classes)
    elif model_flag == "caps":
        model = capsnet.CapsNet(channels, classes)
    elif model_flag == "bpcaps":
        model = bpcapsnet.BPCapsNet(channels, classes)
    elif model_flag == "vit":
        model = MedViT_small(n_channels=channels, num_classes=classes)
    else:
        raise ValueError(f"Unknown model: {model_flag}")

    path = checkpoint_path(model_flag, data_flag)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["net"], strict=True)
    return model.to(device).eval()


def predict_dataset(model, dataset, device, batch_size):
    classifier_model = ClassificationOnlyModel(model).to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    scores = []
    with torch.no_grad():
        for inputs, _ in loader:
            scores.append(classifier_model(inputs.to(device)).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def collect_inputs(dataset, indices):
    return torch.stack([dataset[int(index)][0] for index in indices]).numpy()


def balanced_sample(eligible_indices, targets, count, seed, classes):
    """Sample deterministically with allocations as even as availability permits."""
    eligible_indices = np.asarray(eligible_indices, dtype=np.int64)
    if count <= 0 or count > len(eligible_indices):
        if count > len(eligible_indices):
            raise ValueError(
                f"Requested {count} cases, but only {len(eligible_indices)} are eligible."
            )
        selected = np.sort(eligible_indices)
        allocations = np.bincount(targets[selected], minlength=classes)
        return selected, allocations

    candidates = [eligible_indices[targets[eligible_indices] == label]
                  for label in range(classes)]
    allocations = np.zeros(classes, dtype=np.int64)
    for _ in range(count):
        available = [
            label for label in range(classes)
            if allocations[label] < len(candidates[label])
        ]
        if not available:
            raise ValueError("Not enough eligible examples to fill the cohort.")
        label = min(available, key=lambda value: (allocations[value], value))
        allocations[label] += 1

    rng = np.random.default_rng(seed)
    selected = []
    for label, allocation in enumerate(allocations):
        if allocation:
            selected.append(
                rng.choice(candidates[label], size=allocation, replace=False)
            )
    return np.sort(np.concatenate(selected)), allocations


def manifest_path(output_dir, data_flag, count, seed):
    return output_dir / f"{data_flag}_shared_correct_{count}_seed_{seed}.npz"


def load_shared_manifest(path, args, data_flag):
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared cohort manifest: {path}")

    with np.load(path, allow_pickle=False) as saved:
        manifest = {key: saved[key].copy() for key in saved.files}

    required = {
        "dataset", "models", "seed", "requested_count", "selected_indices",
        "selected_targets", "all_model_predictions",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Manifest {path} is missing fields: {missing}")

    saved_dataset = str(manifest["dataset"].item())
    saved_models = manifest["models"].astype(str).tolist()
    saved_seed = int(manifest["seed"].item())
    saved_count = int(manifest["requested_count"].item())
    missing_models = [model for model in args.models if model not in saved_models]

    if saved_dataset != data_flag:
        raise ValueError(
            f"Manifest {path} is for {saved_dataset}, not {data_flag}."
        )
    if missing_models:
        raise ValueError(
            f"Manifest {path} does not contain models {missing_models}; "
            f"available models are {saved_models}."
        )
    if saved_seed != args.seed:
        raise ValueError(
            f"Manifest {path} uses seed {saved_seed}, not {args.seed}."
        )
    if saved_count != args.num_cases:
        raise ValueError(
            f"Manifest {path} contains {saved_count} requested cases, "
            f"not {args.num_cases}."
        )
    if len(manifest["selected_indices"]) != args.num_cases:
        raise ValueError(
            f"Manifest {path} has {len(manifest['selected_indices'])} selected "
            f"indices, not {args.num_cases}."
        )
    if manifest["all_model_predictions"].shape[0] != len(saved_models):
        raise ValueError(
            f"Manifest {path} prediction rows do not match its model list."
        )

    print(f"Reusing shared cohort manifest: {path}", flush=True)
    return manifest


def build_or_load_shared_manifest(args, data_flag, device):
    if args.num_cases <= 0:
        raise ValueError("shared-correct selection requires --num-cases > 0.")
    if args.shared_manifest is not None:
        return load_shared_manifest(args.shared_manifest, args, data_flag)

    path = manifest_path(args.output_dir, data_flag, args.num_cases, args.seed)
    if path.exists():
        return load_shared_manifest(path, args, data_flag)

    classes = len(INFO[data_flag]["label"])
    all_predictions = []
    targets = None
    for model_flag in args.models:
        dataset = get_dataset(data_flag, model_flag, args.download)
        current_targets = dataset_targets(dataset)
        if targets is None:
            targets = current_targets
        elif not np.array_equal(targets, current_targets):
            raise RuntimeError("Model-specific datasets do not share identical labels.")
        channels = dataset[0][0].shape[0]
        model = load_model(model_flag, data_flag, channels, classes, device)
        scores = predict_dataset(
            model, dataset, device, args.prediction_batch_size
        )
        predictions = np.argmax(scores, axis=1)
        all_predictions.append(predictions)
        print(
            f"{data_flag} {model_flag}: clean argmax accuracy "
            f"{np.mean(predictions == targets):.4f}", flush=True
        )
        del model, dataset, scores
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_predictions = np.asarray(all_predictions, dtype=np.int64)
    shared_correct = np.all(all_predictions == targets[None, :], axis=0)
    eligible = np.flatnonzero(shared_correct)
    selected, allocations = balanced_sample(
        eligible, targets, args.num_cases, args.seed, classes
    )
    np.savez_compressed(
        path,
        dataset=data_flag,
        models=np.asarray(args.models),
        seed=args.seed,
        requested_count=args.num_cases,
        test_set_size=len(targets),
        shared_correct_count=len(eligible),
        selected_indices=selected,
        selected_targets=targets[selected],
        class_allocations=allocations,
        all_model_predictions=all_predictions,
    )
    print(
        f"Saved {path}: {len(eligible)} jointly correct candidates; "
        f"selected class counts {allocations.tolist()}", flush=True
    )
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def load_fixed_cohort_manifest(path, data_flag):
    if not path.is_file():
        raise FileNotFoundError(f"Missing cohort manifest: {path}")
    with np.load(path, allow_pickle=False) as saved:
        required = {"dataset", "test_set_size", "selected_indices", "selected_targets"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                f"Cohort manifest {path} is missing fields: {sorted(missing)}"
            )
        saved_dataset = str(saved["dataset"].item())
        if saved_dataset != data_flag:
            raise ValueError(
                f"Cohort manifest {path} is for {saved_dataset}, not {data_flag}."
            )
        manifest = {key: saved[key].copy() for key in saved.files}
    print(f"Reusing fixed cohort manifest: {path}", flush=True)
    return manifest


def distortion_stats(l2, successful):
    values = l2[successful]
    if not len(values):
        return dict(median_l2=np.nan, q1_l2=np.nan, q3_l2=np.nan,
                    mean_l2=np.nan, std_l2=np.nan)
    q1, median, q3 = np.percentile(values, (25, 50, 75))
    return dict(
        median_l2=float(median), q1_l2=float(q1), q3_l2=float(q3),
        mean_l2=float(np.mean(values)), std_l2=float(np.std(values)),
    )


def result_name(data_flag, model_flag, initial_const):
    label = f"{initial_const:g}".replace(".", "p").replace("-", "m")
    return f"{data_flag}_{model_flag}_cw_const_{label}_metrics.npz"


def run_attack(args, data_flag, model_flag, device, shared_manifest=None):
    output_path = args.output_dir / result_name(
        data_flag, model_flag, args.initial_const
    )
    if output_path.exists():
        print(f"Skipping existing result: {output_path}", flush=True)
        return

    dataset = get_dataset(data_flag, model_flag, args.download)
    targets = dataset_targets(dataset)
    classes = len(INFO[data_flag]["label"])
    channels = dataset[0][0].shape[0]
    model = load_model(model_flag, data_flag, channels, classes, device)

    source_cohort_count = None
    excluded_clean_incorrect_count = 0
    if shared_manifest is not None:
        selected = shared_manifest["selected_indices"].astype(np.int64)
        manifest_test_set_size = int(shared_manifest["test_set_size"].item())
        if manifest_test_set_size != len(targets):
            raise ValueError(
                f"Cohort manifest test-set size is {manifest_test_set_size}, "
                f"but the dataset has {len(targets)} cases."
            )
        if np.any(selected < 0) or np.any(selected >= len(targets)):
            raise ValueError("Cohort manifest contains out-of-range test indices.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Cohort manifest contains duplicate test indices.")
        manifest_targets = shared_manifest["selected_targets"].astype(np.int64)
        if not np.array_equal(manifest_targets, targets[selected]):
            raise ValueError("Cohort manifest targets do not match the dataset.")

        saved_models = (
            shared_manifest.get("models", np.asarray([])).astype(str).tolist()
        )
        if model_flag in saved_models and "all_model_predictions" in shared_manifest:
            model_position = saved_models.index(model_flag)
            full_predictions = shared_manifest["all_model_predictions"][model_position]
        else:
            full_scores = predict_dataset(
                model, dataset, device, args.prediction_batch_size
            )
            full_predictions = np.argmax(full_scores, axis=1)
            del full_scores
        full_clean_correct_count = int(np.sum(full_predictions == targets))
        if args.selection == "fixed-cohort":
            source_cohort_count = len(selected)
            clean_correct = full_predictions[selected] == targets[selected]
            excluded_clean_incorrect_count = int(np.sum(~clean_correct))
            selected = selected[clean_correct]
            if not len(selected):
                raise ValueError(
                    f"No clean-correct {model_flag} cases remain in the fixed cohort."
                )
            print(
                f"Fixed cohort for {model_flag}: attacking {len(selected)}/"
                f"{source_cohort_count} clean-correct cases; excluded "
                f"{excluded_clean_incorrect_count} clean errors",
                flush=True,
            )
            selection = "fixed-cohort-clean-correct"
        else:
            source_cohort_count = len(selected)
            selection = "shared-correct"
    else:
        full_scores = predict_dataset(
            model, dataset, device, args.prediction_batch_size
        )
        full_predictions = np.argmax(full_scores, axis=1)
        eligible = np.flatnonzero(full_predictions == targets)
        selected, _ = balanced_sample(
            eligible, targets, args.num_cases, args.seed, classes
        )
        full_clean_correct_count = len(eligible)
        selection = "model-correct"
        del full_scores

    if source_cohort_count is None:
        source_cohort_count = len(selected)

    selected_inputs = collect_inputs(dataset, selected)
    selected_targets = targets[selected]
    classifier = create_attack_classifier(
        model, selected_inputs.shape[1:], classes, device
    )
    clean_scores = classifier.predict(
        selected_inputs, batch_size=args.batch_size
    )
    clean_predictions = np.argmax(clean_scores, axis=1)
    if not np.all(clean_predictions == selected_targets):
        bad = selected[clean_predictions != selected_targets]
        raise RuntimeError(
            f"Selected cohort is not clean-correct for {model_flag}: {bad.tolist()}"
        )

    print(
        f"Running {data_flag} {model_flag}: {len(selected)} cases, "
        f"initial_const={args.initial_const:g}", flush=True
    )
    attacker = CarliniL2Method(
        classifier=classifier,
        confidence=args.confidence,
        targeted=False,
        learning_rate=args.learning_rate,
        binary_search_steps=args.binary_search_steps,
        max_iter=args.max_iter,
        initial_const=args.initial_const,
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
        ((adversarial_inputs - selected_inputs) / 2.0).reshape(len(selected), -1),
        axis=1,
    )
    stats = distortion_stats(l2, successful)

    np.savez_compressed(
        output_path,
        dataset=data_flag,
        model=model_flag,
        selection=selection,
        initial_const=args.initial_const,
        num_cases=len(selected),
        initially_correct_count=len(selected),
        test_set_size=len(targets),
        full_clean_correct_count=full_clean_correct_count,
        full_clean_accuracy=full_clean_correct_count / len(targets),
        success_count=int(np.sum(successful)),
        failure_count=int(len(selected) - np.sum(successful)),
        asr=float(np.mean(successful)),
        robust_accuracy_on_attacked_cohort=float(1.0 - np.mean(successful)),
        max_iter=args.max_iter,
        binary_search_steps=args.binary_search_steps,
        learning_rate=args.learning_rate,
        confidence=args.confidence,
        seed=args.seed,
        source_cohort_count=source_cohort_count,
        excluded_clean_incorrect_count=excluded_clean_incorrect_count,
        input_shape=np.asarray(selected_inputs.shape[1:], dtype=np.int64),
        l2_space="model_input_[0,1]",
        selected_indices=selected,
        targets=selected_targets,
        clean_scores=clean_scores,
        clean_predictions=clean_predictions,
        adversarial_scores=adversarial_scores,
        adversarial_predictions=adversarial_predictions,
        successful=successful,
        l2=l2,
        successful_l2=l2[successful],
        **stats,
    )
    print(
        f"Saved {output_path}: ASR={np.mean(successful):.4f} "
        f"({np.sum(successful)}/{len(selected)}), "
        f"median L2={stats['median_l2']:.6f}", flush=True
    )

    del attacker, classifier, model, dataset, selected_inputs, adversarial_inputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def write_summary(output_dir):
    fields = [
        "dataset", "model", "selection", "initial_const", "num_cases",
        "test_set_size", "full_clean_correct_count", "full_clean_accuracy",
        "success_count", "failure_count", "asr", "median_l2", "q1_l2",
        "q3_l2", "mean_l2", "std_l2", "max_iter",
        "binary_search_steps", "learning_rate", "confidence", "seed",
    ]
    rows = []
    for path in sorted(output_dir.glob("*_metrics.npz")):
        with np.load(path, allow_pickle=False) as result:
            rows.append({field: result[field].item() for field in fields})
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args):
    if args.num_cases < 0:
        raise ValueError("--num-cases cannot be negative.")
    if args.initial_const <= 0 or args.learning_rate <= 0:
        raise ValueError("CW constant and learning rate must be positive.")
    if args.max_iter <= 0 or args.binary_search_steps <= 0:
        raise ValueError("Iteration counts must be positive.")
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models cannot contain duplicates.")
    if args.selection == "shared-correct" and len(args.datasets) != 1:
        raise ValueError("shared-correct selection supports one dataset per run.")
    if args.selection == "fixed-cohort":
        if args.cohort_manifest is None:
            raise ValueError("fixed-cohort selection requires --cohort-manifest.")
        if len(args.datasets) != 1:
            raise ValueError("fixed-cohort selection supports one dataset per run.")
    elif args.cohort_manifest is not None:
        raise ValueError("--cohort-manifest requires --selection fixed-cohort.")
    if args.shared_manifest is not None and args.selection != "shared-correct":
        raise ValueError("--shared-manifest requires --selection shared-correct.")


def main():
    args = parse_args()
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"Using device: {device}", flush=True)

    for data_flag in args.datasets:
        shared_manifest = None
        if args.selection == "shared-correct":
            shared_manifest = build_or_load_shared_manifest(
                args, data_flag, device
            )
        elif args.selection == "fixed-cohort":
            shared_manifest = load_fixed_cohort_manifest(
                args.cohort_manifest, data_flag
            )
        for model_flag in args.models:
            run_attack(args, data_flag, model_flag, device, shared_manifest)
            write_summary(args.output_dir)


if __name__ == "__main__":
    main()
