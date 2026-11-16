#!/usr/bin/env python3
"""Benchmark model-only inference latency on a real held-out test batch.

The input batch is prepared with the same transforms used by the training and
evaluation scripts, then moved to the selected device before timing begins.
Consequently, the reported values exclude dataset I/O, preprocessing, and
host-to-device transfer. Capsule models are measured both end to end (including
the reconstruction decoder) and along a classification-only path that stops
after computing capsule class scores.
"""

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as torchvision_datasets
import torchvision.transforms as transforms
from medmnist import INFO
from PIL import Image

from models import bpcapsnet, capsnet
from models.MedViT import MedViT_small
from models.resnets import ResNet18, ResNet50

DATASETS = (
    "mnist",
    "pneumoniamnist",
    "breastmnist",
    "nodulemnist3d",
    "bloodmnist",
)
MODELS = ("resnet18", "resnet50", "vit", "caps", "bpcaps")
CAPSULE_MODELS = ("caps", "bpcaps")


class ToTensor3D:
    """Match the repository's MedMNIST3D slice-to-channel transform."""

    def __call__(self, sample):
        sample = sample.squeeze(0)
        sample = np.transpose(sample, (2, 0, 1))
        return torch.tensor(sample)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--device", default="cuda:0",
        help="PyTorch device used for timing (default: cuda:0)",
    )
    parser.add_argument("--seed", type=int, default=665)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False,
        help="Download missing test data (default: disabled)",
    )
    parser.add_argument(
        "--capsule-mode",
        choices=("both", "end-to-end", "classification-only"),
        default="both",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(dataset, model):
    resize_tag = "_224" if model == "vit" else ""
    return ROOT / "checkpoints" / (
        "{}_{}{}_best_model.pth".format(model, dataset, resize_tag)
    )


def test_dataset(dataset, model, download):
    resize = model == "vit"
    if dataset == "mnist":
        transform_steps = []
        if resize:
            transform_steps.append(
                transforms.Resize((224, 224), interpolation=Image.NEAREST)
            )
        transform_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return torchvision_datasets.MNIST(
            root=str(ROOT / "data"), train=False,
            transform=transforms.Compose(transform_steps),
            download=download,
        )

    info = INFO[dataset]
    dataset_class = getattr(medmnist, info["python_class"])
    if "3d" in dataset:
        transform_steps = [ToTensor3D()]
        if resize:
            transform_steps.append(
                transforms.Resize((224, 224), interpolation=Image.NEAREST)
            )
        transform_steps.extend([
            transforms.Lambda(lambda image: image.float()),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        transform_steps = []
        if resize:
            transform_steps.append(
                transforms.Resize((224, 224), interpolation=Image.NEAREST)
            )
        transform_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    return dataset_class(
        split="test",
        transform=transforms.Compose(transform_steps),
        download=download,
        as_rgb=False,
        root=str(ROOT / "data"),
    )


def load_test_batch(dataset, model, batch_size, download, seed):
    benchmark_dataset = test_dataset(dataset, model, download)
    if len(benchmark_dataset) < batch_size:
        raise ValueError(
            "Test split has {} examples, fewer than batch size {}".format(
                len(benchmark_dataset), batch_size
            )
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(benchmark_dataset), generator=generator)[
        :batch_size
    ].tolist()
    loader = data.DataLoader(
        data.Subset(benchmark_dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    inputs, _ = next(iter(loader))
    return inputs, indices, len(benchmark_dataset)


def class_count(dataset):
    return 10 if dataset == "mnist" else len(INFO[dataset]["label"])


def build_model(model_name, channels, classes):
    if model_name == "resnet18":
        return ResNet18(in_channels=channels, num_classes=classes)
    if model_name == "resnet50":
        return ResNet50(in_channels=channels, num_classes=classes)
    if model_name == "vit":
        return MedViT_small(n_channels=channels, num_classes=classes)
    if model_name == "caps":
        return capsnet.CapsNet(channels, classes)
    if model_name == "bpcaps":
        return bpcapsnet.BPCapsNet(channels, classes)
    raise ValueError("Unsupported model: {}".format(model_name))


class CapsuleClassificationOnly(nn.Module):
    """Skip reconstruction while preserving the model's class-score path."""

    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name

    def forward(self, inputs):
        outputs = self.model.relu(self.model.conv(inputs))
        outputs = self.model.primary_caps(outputs)
        if self.model_name == "caps":
            outputs = self.model.digit_caps(outputs)
        else:
            outputs, _, _ = self.model.routing_module(outputs)
        return torch.norm(outputs, dim=-1)


def percentile(values, percentile_value):
    return float(np.percentile(np.asarray(values), percentile_value))


def timed_forward(module, inputs, warmup_iterations, iterations, device):
    module.eval()
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            module(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            for start, end in zip(starts, ends):
                start.record()
                module(inputs)
                end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        else:
            elapsed_ms = []
            for _ in range(iterations):
                start = time.perf_counter()
                module(inputs)
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(elapsed_ms)
    return {
        "mean_latency_ms": mean_ms,
        "median_latency_ms": statistics.median(elapsed_ms),
        "p05_latency_ms": percentile(elapsed_ms, 5),
        "p95_latency_ms": percentile(elapsed_ms, 95),
        "min_latency_ms": min(elapsed_ms),
        "max_latency_ms": max(elapsed_ms),
        "throughput_images_per_second": inputs.shape[0] * 1000.0 / mean_ms,
    }


def device_metadata(device):
    metadata = {"device": str(device), "device_type": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update({
            "device_name": properties.name,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "total_device_memory_bytes": properties.total_memory,
        })
    else:
        metadata.update({
            "device_name": platform.processor() or platform.machine(),
            "torch_num_threads": torch.get_num_threads(),
        })
    return metadata


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(path)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if args.iterations < 2:
        raise ValueError("--iterations must be at least 2")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; pass --device cpu to benchmark CPU"
        )
    setup_seed(args.seed)

    inputs, sample_indices, test_size = load_test_batch(
        args.dataset, args.model, args.batch_size, args.download, args.seed
    )
    channels = inputs.shape[1]
    model = build_model(args.model, channels, class_count(args.dataset))
    selected_checkpoint = args.checkpoint or checkpoint_path(args.dataset, args.model)
    if not selected_checkpoint.is_file():
        raise FileNotFoundError("Missing checkpoint: {}".format(selected_checkpoint))
    checkpoint = torch.load(selected_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["net"], strict=True)
    model = model.to(device).eval()
    inputs = inputs.to(device)

    modules = []
    if args.model in CAPSULE_MODELS:
        if args.capsule_mode in ("both", "end-to-end"):
            modules.append(("end_to_end_with_reconstruction", model))
        if args.capsule_mode in ("both", "classification-only"):
            modules.append((
                "classification_only",
                CapsuleClassificationOnly(model, args.model).to(device).eval(),
            ))
    else:
        modules.append(("classification", model))

    if args.model in CAPSULE_MODELS and len(modules) == 2:
        with torch.inference_mode():
            full_scores = modules[0][1](inputs)[0]
            classification_scores = modules[1][1](inputs)
        if not torch.allclose(full_scores, classification_scores):
            raise RuntimeError(
                "Classification-only capsule path does not match full forward scores"
            )

    measurements = []
    for mode, module in modules:
        print("Benchmarking {} / {} / {}...".format(
            args.dataset, args.model, mode
        ), flush=True)
        measurement = timed_forward(
            module, inputs, args.warmup_iterations, args.iterations, device
        )
        measurement["mode"] = mode
        measurements.append(measurement)
        print(
            "  median={:.3f} ms, p95={:.3f} ms, throughput={:.1f} images/s".format(
                measurement["median_latency_ms"],
                measurement["p95_latency_ms"],
                measurement["throughput_images_per_second"],
            ),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": str(selected_checkpoint.resolve()),
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "seed": args.seed,
        "input_source": "real_test_examples",
        "timing_scope": "model_only_excluding_io_preprocessing_and_device_transfer",
        "test_split_size": test_size,
        "sample_indices": sample_indices,
        "input_shape": list(inputs.shape),
        "input_dtype": str(inputs.dtype),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "torch_version": torch.__version__,
        "measurements": measurements,
    }
    payload.update(device_metadata(device))
    write_json(args.output, payload)
    print("Saved {}".format(args.output), flush=True)


if __name__ == "__main__":
    main()
