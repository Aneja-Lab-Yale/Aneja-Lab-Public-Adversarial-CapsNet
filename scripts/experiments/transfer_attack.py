#!/usr/bin/env python3
"""Generate a white-box adversarial test set and evaluate transfer attacks.

The default workflow creates BloodMNIST PGD examples on ResNet-18 at
epsilon=0.016, then evaluates those exact examples on every other model.
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import medmnist
import numpy as np
import torch
import torch.nn.functional as functional
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as torchvision_datasets
from medmnist import INFO

from models import bpcapsnet as bpcaps
from models import capsnet as caps
from models.MedViT import MedViT_small
from models.resnets import ResNet18, ResNet50
from scripts.experiments.sensitivity import (
    create_attack_classifier,
    create_attacker,
    test_classifier,
)
from utils import MNISTEvaluator, ToTensor3D

DATA_ROOT = ROOT / "data"
TRANSFER_DATA_DIR = ROOT / "transfer_attack_data"
TRANSFER_RESULT_DIR = ROOT / "adv_output" / "transfer"
DEFAULT_OUTPUT = (
    TRANSFER_DATA_DIR
    / "bloodmnist_resnet18_pgd_eps_0.016_test.npz"
)
MODEL_FLAGS = ("resnet18", "resnet50", "vit", "caps", "bpcaps")
DEFAULT_TARGETS = ("resnet50", "vit", "caps", "bpcaps")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def select_device(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    if gpu_id >= 0:
        print("CUDA is unavailable; falling back to CPU.")
    return torch.device("cpu")


def dataset_details(data_flag):
    if data_flag == "mnist":
        return None, 1, 10
    info = INFO[data_flag]
    return info, info["n_channels"], len(info["label"])


def native_test_dataset(data_flag, download=False):
    if data_flag == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        return torchvision_datasets.MNIST(
            root=str(DATA_ROOT), train=False, transform=transform,
            download=download,
        )

    info, _, _ = dataset_details(data_flag)
    dataset_class = getattr(medmnist, info["python_class"])
    if "3d" in data_flag:
        transform = transforms.Compose(
            [
                ToTensor3D(),
                transforms.Lambda(lambda image: image.float()),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
    return dataset_class(
        split="test",
        transform=transform,
        download=download,
        as_rgb=False,
        root=str(DATA_ROOT),
    )


def build_model(model_flag, n_channels, n_classes):
    if model_flag == "resnet18":
        return ResNet18(in_channels=n_channels, num_classes=n_classes)
    if model_flag == "resnet50":
        return ResNet50(in_channels=n_channels, num_classes=n_classes)
    if model_flag == "vit":
        return MedViT_small(n_channels=n_channels, num_classes=n_classes)
    if model_flag == "caps":
        return caps.CapsNet(n_channels, n_classes)
    if model_flag == "bpcaps":
        return bpcaps.BPCapsNet(n_channels, n_classes)
    raise ValueError("Unsupported model: {}".format(model_flag))


def default_checkpoint(data_flag, model_flag, checkpoint_dir):
    resize_tag = "_224" if model_flag == "vit" else ""
    return checkpoint_dir / (
        "{}_{}{}_best_model.pth".format(model_flag, data_flag, resize_tag)
    )


def default_transfer_path(data_flag, source_model, epsilon):
    return (
        TRANSFER_DATA_DIR
        / "{}_{}_pgd_eps_{:g}_test.npz".format(
            data_flag, source_model, epsilon
        )
    )


def load_model(model_flag, data_flag, checkpoint, device, n_channels=None):
    _, configured_channels, n_classes = dataset_details(data_flag)
    if n_channels is None:
        n_channels = configured_channels
    model = build_model(model_flag, n_channels, n_classes).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["net"], strict=True)
    model.eval()
    return model


class ResizeInputModel(nn.Module):
    """Apply a differentiable spatial resize before model inference."""

    def __init__(self, model, size=(224, 224)):
        super().__init__()
        self.model = model
        self.size = size

    def forward(self, inputs):
        resized = functional.interpolate(inputs, size=self.size, mode="nearest")
        return self.model(resized)


def generate_archive(args):
    setup_seed(args.seed)
    device = select_device(args.gpu_id)
    dataset = native_test_dataset(args.data_flag, args.download)
    sample_indices = None
    if args.test_subset_size is not None:
        if not 1 <= args.test_subset_size <= len(dataset):
            raise ValueError(
                "--test-subset-size must be between 1 and {}".format(
                    len(dataset)
                )
            )
        generator = np.random.default_rng(args.test_subset_seed)
        sample_indices = np.sort(
            generator.choice(
                len(dataset), size=args.test_subset_size, replace=False
            )
        )
        dataset = data.Subset(dataset, sample_indices.tolist())
        print(
            "Using deterministic test subset: {} images (seed={})".format(
                args.test_subset_size, args.test_subset_seed
            )
        )
    loader = data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False
    )
    _, _, n_classes = dataset_details(args.data_flag)
    checkpoint = args.checkpoint or default_checkpoint(
        args.data_flag, args.source_model, args.checkpoint_dir
    )
    input_shape = tuple(dataset[0][0].shape)
    model = load_model(
        args.source_model,
        args.data_flag,
        checkpoint,
        device,
        n_channels=input_shape[0],
    )
    attack_model = ResizeInputModel(model) if args.source_model == "vit" else model
    classifier = create_attack_classifier(
        attack_model, input_shape, n_classes, device
    )
    attacker = create_attacker("pgd", classifier, args.epsilon)

    clean_batches = []
    adversarial_batches = []
    label_batches = []
    for batch_number, (inputs, labels) in enumerate(loader, start=1):
        clean = inputs.numpy()
        # Match the repository's existing PGD sensitivity protocol: ART uses
        # the source model's predictions when labels are not supplied.
        adversarial = attacker.generate(x=clean)
        clean_batches.append(clean)
        adversarial_batches.append(adversarial)
        label_batches.append(labels.numpy().reshape(-1))
        print(
            "Generated batch {}/{} ({} samples)".format(
                batch_number, len(loader), len(inputs)
            )
        )

    clean_images = np.concatenate(clean_batches).astype(np.float32, copy=False)
    adversarial_images = np.concatenate(adversarial_batches).astype(
        np.float32, copy=False
    )
    labels = np.concatenate(label_batches).astype(np.int64, copy=False)
    max_linf = float(np.max(np.abs(adversarial_images - clean_images)))
    tolerance = max(1e-6, args.epsilon * 1e-5)
    if max_linf > args.epsilon + tolerance:
        raise ValueError(
            "Generated perturbation exceeds epsilon: {:.8f} > {:.8f}".format(
                max_linf, args.epsilon
            )
        )

    output = args.output or default_transfer_path(
        args.data_flag, args.source_model, args.epsilon
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = dict(
        adversarial_images=adversarial_images,
        clean_images=clean_images,
        labels=labels,
        data_flag=np.array(args.data_flag),
        source_model=np.array(args.source_model),
        attack=np.array("pgd"),
        epsilon=np.array(args.epsilon, dtype=np.float64),
        normalized_range=np.array([-1.0, 1.0], dtype=np.float32),
        max_linf=np.array(max_linf, dtype=np.float64),
        seed=np.array(args.seed, dtype=np.int64),
        test_subset_seed=np.array(args.test_subset_seed, dtype=np.int64),
        source_resize=np.array(args.source_model == "vit"),
    )
    if sample_indices is not None:
        archive["sample_indices"] = sample_indices.astype(np.int64, copy=False)
    np.savez_compressed(output, **archive)
    print("Saved {} adversarial test examples to {}".format(len(labels), output))
    print("Maximum normalized-space L-inf perturbation: {:.8f}".format(max_linf))


class TransferDataset(data.Dataset):
    def __init__(self, images, labels, resize=False, scalar_labels=False):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels).long()
        self.resize = resize
        self.scalar_labels = scalar_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.resize:
            image = functional.interpolate(
                image.unsqueeze(0), size=(224, 224), mode="nearest"
            ).squeeze(0)
        label = self.labels[index]
        return image, label if self.scalar_labels else label.reshape(1)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "adversarial_images",
            "labels",
            "data_flag",
            "source_model",
            "attack",
            "epsilon",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                "Transfer archive is missing: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        result = {key: archive[key] for key in archive.files}

    images = result["adversarial_images"]
    labels = result["labels"].reshape(-1)
    if images.ndim != 4 or images.shape[0] != labels.shape[0]:
        raise ValueError(
            "Expected NCHW images and one label per image; got {} and {}".format(
                images.shape, labels.shape
            )
        )
    if not np.isfinite(images).all() or images.min() < -1.00001 or images.max() > 1.00001:
        raise ValueError("Adversarial images must be finite and within [-1, 1].")
    return result


def evaluate_archive(args):
    setup_seed(args.seed)
    device = select_device(args.gpu_id)
    archive = load_archive(args.input)
    data_flag = str(archive["data_flag"].item())
    source_model = str(archive["source_model"].item())
    images = archive["adversarial_images"].astype(np.float32, copy=False)
    labels = archive["labels"].reshape(-1).astype(np.int64, copy=False)
    _, _, n_classes = dataset_details(data_flag)
    if data_flag == "mnist":
        evaluator = MNISTEvaluator("test", labels, root=str(DATA_ROOT))
    else:
        evaluator = medmnist.Evaluator(data_flag, "test", root=str(DATA_ROOT))

    if source_model in args.models:
        print("Note: evaluating the source model as explicitly requested.")

    auc_scores = []
    accuracy_scores = []
    for model_flag in args.models:
        checkpoint = default_checkpoint(data_flag, model_flag, args.checkpoint_dir)
        model = load_model(
            model_flag,
            data_flag,
            checkpoint,
            device,
            n_channels=images.shape[1],
        )
        resize = model_flag == "vit"
        transfer_dataset = TransferDataset(
            images, labels, resize=resize, scalar_labels=data_flag == "mnist"
        )
        loader = data.DataLoader(
            transfer_dataset, batch_size=args.batch_size, shuffle=False
        )
        input_shape = (images.shape[1], 224, 224) if resize else images.shape[1:]
        classifier = create_attack_classifier(
            model, tuple(input_shape), n_classes, device
        )
        auc, accuracy = test_classifier(
            classifier,
            evaluator,
            loader,
            device,
            n_classes,
            args.run,
            mnist=data_flag == "mnist",
        )
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        print(
            "{} transfer performance: AUC={:.6f}, accuracy={:.6f}".format(
                model_flag, auc, accuracy
            )
        )

    output = args.output or TRANSFER_RESULT_DIR / (
        args.input.stem.replace("_test", "") + "_transfer_metrics.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        models=np.asarray(args.models),
        auc=np.asarray(auc_scores, dtype=np.float64),
        accuracy=np.asarray(accuracy_scores, dtype=np.float64),
        data_flag=np.array(data_flag),
        source_model=np.array(source_model),
        attack=archive["attack"],
        epsilon=archive["epsilon"],
        adversarial_set=np.array(str(args.input)),
    )
    print("Saved transfer metrics to {}".format(output))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gpu-id", type=int, default=0)
    common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--seed", type=int, default=665)
    common.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "checkpoints"
    )

    generate = subparsers.add_parser(
        "generate", parents=[common], help="generate and save the source attack"
    )
    generate.add_argument("--data-flag", default="bloodmnist")
    generate.add_argument(
        "--source-model", choices=MODEL_FLAGS, default="resnet18"
    )
    generate.add_argument("--epsilon", type=float, default=0.016)
    generate.add_argument("--checkpoint", type=Path)
    generate.add_argument("--download", action="store_true")
    generate.add_argument("--test-subset-size", type=int)
    generate.add_argument("--test-subset-seed", type=int, default=665)
    generate.add_argument(
        "--output",
        type=Path,
        help="archive path (default: derived from dataset, source, and epsilon)",
    )
    generate.set_defaults(func=generate_archive)

    evaluate = subparsers.add_parser(
        "evaluate", parents=[common], help="evaluate targets on a saved attack"
    )
    evaluate.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    evaluate.add_argument(
        "--models", nargs="+", choices=MODEL_FLAGS, default=list(DEFAULT_TARGETS)
    )
    evaluate.add_argument("--run", default="transfer_resnet18_pgd_eps_0.016")
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(func=evaluate_archive)
    return parser


def main():
    args = build_parser().parse_args()
    if hasattr(args, "epsilon") and args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
