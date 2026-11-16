#!/usr/bin/env python3
"""Generate paired clean/PGD Grad-CAM maps for the interpretability figure.

The default run covers all five models and all four MedMNIST datasets.  It
uses the model-specific PGD archives produced by ``transfer_attack.py`` so
that each clean map is paired with the exact adversarial example generated
against that model.  One resumable NPZ file is written per dataset/model.
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from medmnist import INFO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import bpcapsnet as bpcaps
from models import capsnet as caps
from models.MedViT import MedViT_small
from models.resnets import ResNet18, ResNet50

DEFAULT_DATASETS = (
    "bloodmnist",
    "breastmnist",
    "pneumoniamnist",
    "nodulemnist3d",
)
DEFAULT_MODELS = ("resnet18", "resnet50", "vit", "caps", "bpcaps")
MODEL_DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vit": "MedViT",
    "caps": "CapsNet",
    "bpcaps": "BP-CapsNet",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--models", nargs="+", choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--epsilon", type=float, default=0.016)
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Grad-CAM batch size; reduce this if MedViT exhausts GPU memory",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument("--seed", type=int, default=665)
    parser.add_argument(
        "--target",
        choices=("predicted", "label", "clean-predicted"),
        default="predicted",
        help=(
            "Class explained by each map. 'predicted' matches the old pipeline; "
            "'clean-predicted' uses the clean prediction for both paired maps."
        ),
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Optional prefix of the test set, useful for smoke tests",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "checkpoints",
    )
    parser.add_argument(
        "--attack-data-dir", type=Path, default=ROOT / "transfer_attack_data",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: gradcam_data/pgd_eps_<epsilon>",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate existing outputs",
    )
    return parser.parse_args()


def epsilon_tag(epsilon):
    return "{:g}".format(epsilon)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def archive_path(directory, dataset, model_name, epsilon):
    return directory / "{}_{}_pgd_eps_{}_test.npz".format(
        dataset, model_name, epsilon_tag(epsilon)
    )


def checkpoint_path(directory, dataset, model_name):
    resize_tag = "_224" if model_name == "vit" else ""
    return directory / "{}_{}{}_best_model.pth".format(
        model_name, dataset, resize_tag
    )


def build_model(model_name, n_channels, n_classes):
    if model_name == "resnet18":
        return ResNet18(in_channels=n_channels, num_classes=n_classes)
    if model_name == "resnet50":
        return ResNet50(in_channels=n_channels, num_classes=n_classes)
    if model_name == "vit":
        return MedViT_small(n_channels=n_channels, num_classes=n_classes)
    if model_name == "caps":
        return caps.CapsNet(n_channels, n_classes)
    if model_name == "bpcaps":
        return bpcaps.BPCapsNet(n_channels, n_classes)
    raise ValueError("Unsupported model: {}".format(model_name))


class GradCamModel(nn.Module):
    """Normalize model inputs and outputs for pytorch-grad-cam."""

    def __init__(self, model, resize=False, capsule=False):
        super().__init__()
        self.model = model
        self.resize = resize
        self.capsule = capsule

    def forward(self, inputs):
        if self.resize:
            inputs = functional.interpolate(
                inputs, size=(224, 224), mode="nearest"
            )
        output = self.model(inputs)
        return output[0] if self.capsule else output


def target_layer(model, model_name):
    # Preserve the layer choices used by the repository's original Grad-CAM.
    if model_name.startswith("resnet"):
        return model.layer4, "layer4"
    if model_name == "vit":
        return model.features[18], "features.18"
    if model_name in ("caps", "bpcaps"):
        return model.conv, "conv"
    raise ValueError("Unsupported model: {}".format(model_name))


def load_archive(path, dataset, model_name, epsilon, max_samples):
    if not path.is_file():
        raise FileNotFoundError("Missing PGD archive: {}".format(path))
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "clean_images", "adversarial_images", "labels", "data_flag",
            "source_model", "attack", "epsilon",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError("{} is missing keys: {}".format(path, sorted(missing)))
        clean = saved["clean_images"].astype(np.float32, copy=False)
        adversarial = saved["adversarial_images"].astype(np.float32, copy=False)
        labels = saved["labels"].reshape(-1).astype(np.int64, copy=False)
        stored_dataset = str(saved["data_flag"].item())
        stored_model = str(saved["source_model"].item())
        stored_attack = str(saved["attack"].item())
        stored_epsilon = float(saved["epsilon"].item())

    if stored_dataset != dataset or stored_model != model_name:
        raise ValueError("Archive metadata does not match {} / {}".format(
            dataset, model_name
        ))
    if stored_attack.lower() != "pgd" or not np.isclose(stored_epsilon, epsilon):
        raise ValueError("Expected PGD epsilon {} in {}".format(epsilon, path))
    if clean.shape != adversarial.shape or len(clean) != len(labels):
        raise ValueError("Mismatched clean/adversarial/label shapes in {}".format(path))
    if clean.ndim != 4 or not np.isfinite(clean).all() or not np.isfinite(adversarial).all():
        raise ValueError("Images must be finite NCHW arrays in {}".format(path))
    max_linf = float(np.max(np.abs(adversarial - clean)))
    if max_linf > epsilon + max(1e-6, epsilon * 1e-5):
        raise ValueError("PGD budget exceeded in {}: {} > {}".format(
            path, max_linf, epsilon
        ))
    total_samples = len(labels)
    stop = total_samples if max_samples is None else min(max_samples, total_samples)
    indices = np.arange(stop, dtype=np.int64)
    return (
        clean[:stop], adversarial[:stop], labels[:stop], indices, max_linf,
        total_samples,
    )


def predict(model, images, batch_size, device):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            inputs = torch.from_numpy(images[start:start + batch_size]).to(device)
            predictions.append(model(inputs).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64, copy=False)


def generate_maps(model, target, images, target_classes, batch_size, device):
    maps = []
    model.eval()
    with GradCAM(model=model, target_layers=[target]) as cam:
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            inputs = torch.from_numpy(images[start:stop]).to(device)
            targets = [ClassifierOutputTarget(int(category))
                       for category in target_classes[start:stop]]
            batch_maps = cam(
                input_tensor=inputs,
                targets=targets,
                aug_smooth=False,
                eigen_smooth=False,
            )
            # MedViT runs at 224x224, but maps are stored at the source image
            # resolution so all five model outputs are directly comparable.
            source_size = images.shape[-2:]
            if batch_maps.shape[-2:] != source_size:
                batch_maps = functional.interpolate(
                    torch.from_numpy(batch_maps).unsqueeze(1),
                    size=source_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).numpy()
            maps.append(batch_maps.astype(np.float32, copy=False))
    return np.concatenate(maps, axis=0)


def process_pair(dataset, model_name, args, device):
    pair_dir = args.output_dir / dataset / model_name
    output = pair_dir / "gradcam_maps.npz"
    archive = archive_path(args.attack_data_dir, dataset, model_name, args.epsilon)
    if output.is_file() and not args.overwrite:
        if not archive.is_file():
            raise FileNotFoundError("Missing PGD archive: {}".format(archive))
        with np.load(archive, allow_pickle=False) as source:
            total_samples = len(source["labels"])
        expected_samples = (
            total_samples if args.max_samples is None
            else min(args.max_samples, total_samples)
        )
        with np.load(output, allow_pickle=False) as saved:
            compatible = (
                str(saved["target_mode"].item()) == args.target
                and np.isclose(float(saved["epsilon"].item()), args.epsilon)
                and len(saved["test_indices"]) == expected_samples
            )
        if compatible:
            print("Reusing {}".format(output), flush=True)
            return
        raise ValueError(
            "Existing output has different settings or sample count: {}. "
            "Pass --overwrite to replace it.".format(output)
        )

    checkpoint = checkpoint_path(args.checkpoint_dir, dataset, model_name)
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint))
    clean, adversarial, labels, indices, max_linf, total_samples = load_archive(
        archive, dataset, model_name, args.epsilon, args.max_samples
    )
    n_classes = len(INFO[dataset]["label"])
    network = build_model(model_name, clean.shape[1], n_classes).to(device)
    state = torch.load(checkpoint, map_location=device)
    network.load_state_dict(state["net"], strict=True)
    cam_model = GradCamModel(
        network, resize=model_name == "vit", capsule="caps" in model_name
    ).to(device)
    layer, layer_name = target_layer(network, model_name)

    print("Generating {} / {} ({} pairs)".format(
        dataset, model_name, len(labels)
    ), flush=True)
    clean_predictions = predict(cam_model, clean, args.batch_size, device)
    adversarial_predictions = predict(
        cam_model, adversarial, args.batch_size, device
    )
    if args.target == "label":
        clean_targets = labels
        adversarial_targets = labels
    elif args.target == "clean-predicted":
        clean_targets = clean_predictions
        adversarial_targets = clean_predictions
    else:
        clean_targets = clean_predictions
        adversarial_targets = adversarial_predictions

    clean_maps = generate_maps(
        cam_model, layer, clean, clean_targets, args.batch_size, device
    )
    adversarial_maps = generate_maps(
        cam_model, layer, adversarial, adversarial_targets,
        args.batch_size, device
    )
    pair_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_output,
        clean_gradcam=clean_maps,
        adversarial_gradcam=adversarial_maps,
        labels=labels,
        clean_predictions=clean_predictions,
        adversarial_predictions=adversarial_predictions,
        clean_targets=np.asarray(clean_targets, dtype=np.int64),
        adversarial_targets=np.asarray(adversarial_targets, dtype=np.int64),
        test_indices=indices,
        total_test_samples=np.array(total_samples, dtype=np.int64),
        dataset=np.array(dataset),
        model=np.array(model_name),
        model_display_name=np.array(MODEL_DISPLAY_NAMES[model_name]),
        target_mode=np.array(args.target),
        target_layer=np.array(layer_name),
        cam_method=np.array("gradcam"),
        cam_normalized_range=np.asarray([0.0, 1.0], dtype=np.float32),
        attack=np.array("pgd"),
        epsilon=np.array(args.epsilon, dtype=np.float64),
        max_linf=np.array(max_linf, dtype=np.float64),
        source_image_shape=np.asarray(clean.shape[1:], dtype=np.int64),
        checkpoint=np.array(str(checkpoint.resolve())),
        attack_archive=np.array(str(archive.resolve())),
    )
    temporary_output.replace(output)
    print("Saved {}".format(output), flush=True)
    del network, cam_model, clean_maps, adversarial_maps
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.epsilon < 0:
        raise ValueError("--epsilon cannot be negative")
    if args.output_dir is None:
        args.output_dir = ROOT / "gradcam_data" / (
            "pgd_eps_{}".format(epsilon_tag(args.epsilon))
        )
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    print("Using device: {}".format(device), flush=True)
    print("Output directory: {}".format(args.output_dir), flush=True)
    for dataset in args.datasets:
        for model_name in args.models:
            process_pair(dataset, model_name, args, device)


if __name__ == "__main__":
    main()
