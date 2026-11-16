# Evaluate models against adversarial attacks
# of varying strength
# Aneja Lab | Yale School of Medicine
# Adapted from https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-Adversarial-Imaging
# Authors: Anand Srinivasan, Marina Joel
# 05/29/25

from art.attacks.evasion import (
    AutoAttack,
    AutoProjectedGradientDescent,
    BasicIterativeMethod,
    CarliniL2Method,
    DeepFool,
    FastGradientMethod,
    ProjectedGradientDescent,
    SquareAttack,
)
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import os


# ADVERSARIAL ATTACK CONFIGS
max_iter = 30
num_random_init = 3
cw_max_iter = 500
# Dataset transforms normalize pixel values from [0, 1] to [-1, 1].
CLIP_VALUES = (-1, 1)


class ClassificationOnlyModel(nn.Module):
    """Expose only class scores, excluding capsule reconstruction outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


class ResizeInputModel(nn.Module):
    """Resize native-resolution inputs inside the differentiable model path."""

    def __init__(self, model, size=(224, 224)):
        super().__init__()
        self.model = model
        self.size = size

    def forward(self, inputs):
        inputs = functional.interpolate(inputs, size=self.size, mode="nearest")
        return self.model(inputs)


def create_attack_classifier(model, input_shape, n_classes, device):
    """Build one classification-only ART estimator for every architecture."""
    attack_model = ClassificationOnlyModel(model)
    attack_model.eval()
    return PyTorchClassifier(
        model=attack_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=n_classes,
        optimizer=None,
        channels_first=True,
        clip_values=CLIP_VALUES,
        device_type='gpu' if device.type == 'cuda' else 'cpu',
    )


def create_attacker(
        attack_type, classifier, eps,
        pgd_max_iter=None, pgd_num_random_init=None):
    """Construct an untargeted classification attack."""
    if attack_type == 'pgd':
        attack_max_iter = max_iter if pgd_max_iter is None else pgd_max_iter
        attack_random_restarts = (
            num_random_init
            if pgd_num_random_init is None
            else pgd_num_random_init
        )
        if attack_max_iter < 1:
            raise ValueError("PGD max_iter must be at least 1")
        if attack_random_restarts < 0:
            raise ValueError("PGD num_random_init cannot be negative")
        return ProjectedGradientDescent(
            classifier,
            eps=eps,
            eps_step=eps / 4,
            max_iter=attack_max_iter,
            num_random_init=attack_random_restarts,
        )
    if attack_type == 'fgsm':
        return FastGradientMethod(classifier, eps=eps)
    if attack_type == 'bim':
        return BasicIterativeMethod(
            classifier,
            eps=eps,
            eps_step=eps / max_iter,
            max_iter=max_iter,
        )
    if attack_type == 'cw':
        return CarliniL2Method(
            classifier=classifier,
            confidence=0.0,
            targeted=False,
            learning_rate=0.01,
            binary_search_steps=10,
            max_iter=cw_max_iter,
            initial_const=0.01,
            batch_size=16,
            verbose=True,
        )
    if attack_type == 'autoattack':
        attacks = None
        if classifier.nb_classes < 3:
            # DLR uses the three largest class scores and is undefined for
            # binary classifiers. Keep the other AutoAttack components.
            attacks = [
                AutoProjectedGradientDescent(
                    estimator=classifier,
                    norm=np.inf,
                    eps=eps,
                    eps_step=eps / 4,
                    max_iter=100,
                    targeted=False,
                    nb_random_init=5,
                    batch_size=16,
                    loss_type='cross_entropy',
                ),
                DeepFool(
                    classifier=classifier,
                    max_iter=100,
                    epsilon=1e-3,
                    nb_grads=classifier.nb_classes,
                    batch_size=16,
                ),
                SquareAttack(
                    estimator=classifier,
                    norm=np.inf,
                    max_iter=5000,
                    eps=eps,
                    p_init=0.8,
                    nb_restarts=5,
                ),
            ]
        return AutoAttack(
            estimator=classifier,
            norm=np.inf,
            eps=eps,
            eps_step=eps / 4,
            attacks=attacks,
            batch_size=16,
            targeted=False,
            parallel=False,
        )
    raise ValueError("Unrecognized attack type: {}".format(attack_type))


def compute_cw_metrics(clean_predictions, adversarial_predictions, targets,
                       clean_inputs, adversarial_inputs):
    """Compute untargeted CW-L2 metrics over initially correct examples.

    L2 distances are reported in the original [0, 1] pixel space. The model
    inputs use [-1, 1], so perturbations are divided by two before measuring
    their norm. Distortion statistics include successful attacks only.
    """
    clean_predictions = np.asarray(clean_predictions).reshape(-1)
    adversarial_predictions = np.asarray(adversarial_predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    clean_inputs = np.asarray(clean_inputs)
    adversarial_inputs = np.asarray(adversarial_inputs)

    sample_count = targets.shape[0]
    if not (
        clean_predictions.shape[0]
        == adversarial_predictions.shape[0]
        == clean_inputs.shape[0]
        == adversarial_inputs.shape[0]
        == sample_count
    ):
        raise ValueError("CW metric inputs must contain the same number of samples.")

    initially_correct = clean_predictions == targets
    successful = initially_correct & (adversarial_predictions != targets)
    initially_correct_count = int(np.sum(initially_correct))
    success_count = int(np.sum(successful))
    failure_count = initially_correct_count - success_count
    asr = (
        success_count / initially_correct_count
        if initially_correct_count > 0 else np.nan
    )

    perturbations = (adversarial_inputs - clean_inputs) / 2.0
    l2_all = np.linalg.norm(perturbations.reshape(sample_count, -1), axis=1)
    successful_l2 = l2_all[successful]

    if success_count > 0:
        q1_l2, median_l2, q3_l2 = np.percentile(
            successful_l2, [25, 50, 75]
        )
        mean_l2 = np.mean(successful_l2)
        std_l2 = np.std(successful_l2)
    else:
        q1_l2 = median_l2 = q3_l2 = mean_l2 = std_l2 = np.nan

    return {
        'asr': float(asr),
        'median_l2': float(median_l2),
        'q1_l2': float(q1_l2),
        'q3_l2': float(q3_l2),
        'mean_l2': float(mean_l2),
        'std_l2': float(std_l2),
        'failure_count': failure_count,
        'success_count': success_count,
        'initially_correct_count': initially_correct_count,
        'total_count': sample_count,
        'successful_l2': successful_l2,
    }


def generate_cw_adversarial_examples(dataloader, attacker, classifier):
    """Run CW-L2 on initially correct inputs and collect reporting metrics."""
    adversarial_batches = []
    label_batches = []
    clean_inputs = []
    adversarial_inputs = []
    clean_predictions = []
    adversarial_predictions = []
    targets_all = []

    for inputs, targets in dataloader:
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy().reshape(-1)
        clean_scores = classifier.predict(inputs_np)
        clean_pred = np.argmax(clean_scores, axis=1)
        initially_correct = clean_pred == targets_np

        # Misclassified clean inputs do not belong in the CW ASR denominator.
        # Leaving them unchanged also avoids spending CW optimization time on
        # examples that are already errors.
        adv_np = inputs_np.copy()
        if np.any(initially_correct):
            adv_np[initially_correct] = attacker.generate(
                x=inputs_np[initially_correct]
            )

        adv_scores = classifier.predict(adv_np)
        adv_pred = np.argmax(adv_scores, axis=1)

        adversarial_batches.append(adv_np)
        label_batches.append(targets)
        clean_inputs.append(inputs_np)
        adversarial_inputs.append(adv_np)
        clean_predictions.append(clean_pred)
        adversarial_predictions.append(adv_pred)
        targets_all.append(targets_np)

    metrics = compute_cw_metrics(
        np.concatenate(clean_predictions),
        np.concatenate(adversarial_predictions),
        np.concatenate(targets_all),
        np.concatenate(clean_inputs),
        np.concatenate(adversarial_inputs),
    )
    return adversarial_batches, label_batches, metrics

def test_classifier(classifier, evaluator, data_loader, device, n_classes, run, mnist=False):

    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach().cpu().numpy()
            outputs = classifier.predict(inputs)
            outputs = torch.from_numpy(outputs).to(device)
            if not mnist:
                targets = torch.squeeze(targets, dim=1) #comment out for mnist
            targets = torch.eye(n_classes).index_select(dim=0, index=targets).to(device)
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, None, run)

        return [auc, acc]

def generate_adversarial_examples(
        dataloader, attacker, device, include_clean=False):
    adversarial_examples = []  # List to store adversarial examples
    labels = []                # List to store labels
    clean_examples = []

    for inputs, targets in dataloader:
        # Move inputs and targets to the appropriate device
        #inputs, targets = inputs.to(device), targets.to(device)

        # Convert inputs to NumPy array (required by ART)
        inputs = inputs.cpu().numpy()
        if include_clean:
            clean_examples.append(inputs)

        # Generate adversarial examples
        adv_examples = attacker.generate(x=inputs)

        # Convert adversarial examples back to PyTorch tensors
        #adv_examples = torch.from_numpy(adv_examples_np).to(device)

        # Save the results
        adversarial_examples.append(adv_examples)
        labels.append(targets)

    if include_clean:
        return adversarial_examples, labels, clean_examples
    return adversarial_examples, labels


def save_transfer_archive(path, adversarial_batches, clean_batches,
                          label_batches, data_flag, source_model, epsilon,
                          seed, source_resize, sample_indices=None):
    """Save sweep-generated examples in the transfer-attack archive format."""
    adversarial_images = np.concatenate(adversarial_batches).astype(
        np.float32, copy=False
    )
    clean_images = np.concatenate(clean_batches).astype(np.float32, copy=False)
    labels = np.concatenate([
        batch.detach().cpu().numpy().reshape(-1) for batch in label_batches
    ]).astype(np.int64, copy=False)
    max_linf = float(np.max(np.abs(adversarial_images - clean_images)))
    tolerance = max(1e-6, epsilon * 1e-5)
    if max_linf > epsilon + tolerance:
        raise ValueError(
            "Generated perturbation exceeds epsilon: {:.8f} > {:.8f}".format(
                max_linf, epsilon
            )
        )

    archive = {
        'adversarial_images': adversarial_images,
        'clean_images': clean_images,
        'labels': labels,
        'data_flag': np.array(data_flag),
        'source_model': np.array(source_model),
        'attack': np.array('pgd'),
        'epsilon': np.array(epsilon, dtype=np.float64),
        'normalized_range': np.array([-1.0, 1.0], dtype=np.float32),
        'max_linf': np.array(max_linf, dtype=np.float64),
        'seed': np.array(seed, dtype=np.int64),
        'test_subset_seed': np.array(seed, dtype=np.int64),
        'source_resize': np.array(source_resize),
    }
    if sample_indices is not None:
        archive['sample_indices'] = np.asarray(sample_indices, dtype=np.int64)

    path = os.fspath(path)
    output_directory = os.path.dirname(path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    np.savez_compressed(path, **archive)
    print("Saved {} adversarial test examples to {}".format(len(labels), path))
    print("Maximum normalized-space L-inf perturbation: {:.8f}".format(max_linf))

def generate_adversarial_examples_from_model_input(
    model, input_shape,
    train_loader, val_loader, test_loader,
    optimizer, criterion, model_flag,
    train_evaluator, val_evaluator, test_evaluator,
    device, run, n_classes, data_flag,
    attack_type='fgsm', eps=0.01):
    robust_classifier = create_attack_classifier(model, input_shape, n_classes, device)
    
    # train_metrics = test_classifier(robust_classifier, train_evaluator, train_loader, device, n_classes, run)
    # val_metrics = test_classifier(robust_classifier, val_evaluator, val_loader, device, n_classes, run)
    # test_metrics = test_classifier(robust_classifier, test_evaluator, test_loader, device, n_classes, run)

    # print("Train acc {}, Val acc {}, Test acc {}".format(train_metrics[1], val_metrics[1], test_metrics[1]))
    # print("Train auc {}, Val auc {}, Test auc {}".format(train_metrics[0], val_metrics[0], test_metrics[0]))
    
    accuracy_scores = []
    auc_scores = []

    if attack_type == 'cw':
        print("Creating CW-L2 attack with max_iter={}.".format(cw_max_iter))
    else:
        print("Creating attacks for eps: {}".format(eps))

    attacker = create_attacker(attack_type, robust_classifier, eps)
        
    return generate_adversarial_examples(test_loader, attacker, device)

def generate_and_test_adversarial(
    model, input_shape,
    train_loader, val_loader, test_loader,
    optimizer, criterion, model_flag,
    train_evaluator, val_evaluator, test_evaluator,
    device, run, n_classes,
    data_flag, attack_type='pgd', resize=False, eps=0.032, output_tag=None,
    attack_resize=False, transfer_archive_path=None,
    transfer_epsilon=0.016, transfer_seed=665, sample_indices=None,
    attack_epsilons=None, pgd_max_iter=None, pgd_num_random_init=None,
    output_subdir=None):
    attack_model = ResizeInputModel(model) if attack_resize else model
    robust_classifier = create_attack_classifier(
        attack_model, input_shape, n_classes, device
    )
    
    is_mnist = True if data_flag == 'mnist' else False
    train_metrics = test_classifier(robust_classifier, train_evaluator, train_loader, device, n_classes, run, is_mnist)
    val_metrics = test_classifier(robust_classifier, val_evaluator, val_loader, device, n_classes, run, is_mnist)
    test_metrics = test_classifier(robust_classifier, test_evaluator, test_loader, device, n_classes, run, is_mnist)

    print("Train acc {}, Val acc {}, Test acc {}".format(train_metrics[1], val_metrics[1], test_metrics[1]))
    print("Train auc {}, Val auc {}, Test auc {}".format(train_metrics[0], val_metrics[0], test_metrics[0]))

    if attack_epsilons is not None:
        if attack_type == 'cw':
            raise ValueError("Explicit attack epsilons are not supported for CW")
        eps_range = [float(value) for value in attack_epsilons]
        if not eps_range or any(value <= 0 for value in eps_range):
            raise ValueError("Attack epsilons must contain positive values")
    elif attack_type == 'cw':
        # CW-L2 minimizes distortion rather than evaluating an epsilon sweep.
        eps_range = [None]
    elif attack_type == 'autoattack':
        # The full AutoAttack ensemble is too expensive for the standard
        # nine-point sweep; use the command-line epsilon for this attack.
        eps_range = [eps]
    else:
        eps_range = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]
    # evaluate sensitivity scores of each image
    eps_scores = [1] * len(test_loader)
    
    accuracy_scores = []
    auc_scores = []

    if output_tag and output_tag.startswith('pgd_at_eps_'):
        save_dir = os.path.join('adv_output', 'adversarial_training')
    else:
        save_dir = os.path.join('adv_output', attack_type)
    if output_subdir:
        normalized_subdir = os.path.normpath(output_subdir)
        if (os.path.isabs(output_subdir)
                or normalized_subdir in ('.', '..')
                or normalized_subdir.startswith('..' + os.sep)):
            raise ValueError("output_subdir must stay within the attack directory")
        save_dir = os.path.join(save_dir, normalized_subdir)
    os.makedirs(save_dir, exist_ok=True)

    # AutoAttack evaluates a single command-line epsilon, so include it in the
    # filename to prevent a new run from silently overwriting another budget.
    attack_label = (
        f"{attack_type}_eps_{eps:g}" if attack_type == 'autoattack' else attack_type
    )

    tagged_model = model_flag
    if resize:
        tagged_model += "_224"
    if output_tag:
        tagged_model += f"_{output_tag}"

    acc_scores_save_path = os.path.join(
        save_dir, f"{data_flag}_{tagged_model}_{attack_label}_acc.npy"
    )
    auc_scores_save_path = os.path.join(
        save_dir, f"{data_flag}_{tagged_model}_{attack_label}_auc.npy"
    )

    cw_metrics_save_path = acc_scores_save_path.replace(
        '_acc.npy', '_metrics.npz'
    )

    for attack_eps in eps_range:
        if attack_type == 'cw':
            print("Creating CW-L2 attack with max_iter={}.".format(cw_max_iter))
        else:
            print("Creating attacks for eps: {}".format(attack_eps))

        attacker = create_attacker(
            attack_type,
            robust_classifier,
            attack_eps,
            pgd_max_iter=pgd_max_iter,
            pgd_num_random_init=pgd_num_random_init,
        )

        if attack_type == 'cw':
            x_test_adv, y_test_adv, cw_metrics = generate_cw_adversarial_examples(
                test_loader, attacker, robust_classifier
            )
            print("Generated CW-L2 adversarial examples.")
        else:
            save_this_set = (
                transfer_archive_path is not None
                and attack_type == 'pgd'
                and np.isclose(attack_eps, transfer_epsilon)
            )
            generated = generate_adversarial_examples(
                test_loader, attacker, device, include_clean=save_this_set
            )
            if save_this_set:
                x_test_adv, y_test_adv, x_test_clean = generated
                save_transfer_archive(
                    transfer_archive_path,
                    x_test_adv,
                    x_test_clean,
                    y_test_adv,
                    data_flag,
                    model_flag,
                    attack_eps,
                    transfer_seed,
                    attack_resize,
                    sample_indices,
                )
            else:
                x_test_adv, y_test_adv = generated
            print("Generated {} adversarial examples for eps {}".format(
                attack_type, attack_eps
            ))

        adv_loader_custom = zip(x_test_adv, y_test_adv)
        adv_metrics = test_classifier(robust_classifier, test_evaluator, adv_loader_custom, device, n_classes, run, is_mnist)
        
        if attack_type == 'cw':
            print("CW-L2: Acc - {} Auc - {}".format(
                adv_metrics[1], adv_metrics[0]
            ))
            print(
                "CW-L2 ASR: {:.4f} ({}/{} initially correct)".format(
                    cw_metrics['asr'], cw_metrics['success_count'],
                    cw_metrics['initially_correct_count']
                )
            )
            print(
                "CW-L2 median L2 [IQR]: {:.6f} [{:.6f}, {:.6f}]".format(
                    cw_metrics['median_l2'], cw_metrics['q1_l2'],
                    cw_metrics['q3_l2']
                )
            )
            print(
                "CW-L2 mean L2 +/- std: {:.6f} +/- {:.6f}".format(
                    cw_metrics['mean_l2'], cw_metrics['std_l2']
                )
            )
            print("CW-L2 failures: {}/{}".format(
                cw_metrics['failure_count'],
                cw_metrics['initially_correct_count']
            ))
            cw_metrics.update({
                'max_iter': cw_max_iter,
                'binary_search_steps': 10,
                'confidence': 0.0,
            })
            np.savez(cw_metrics_save_path, **cw_metrics)
        else:
            print("Eps {}: Acc - {} Auc - {}".format(
                attack_eps, adv_metrics[1], adv_metrics[0]
            ))

        auc_scores.append(adv_metrics[0])
        accuracy_scores.append(adv_metrics[1])
        #np.save(acc_scores_save_path, np.array(accuracy_scores, dtype='float'))
    
    np.save(acc_scores_save_path, np.array(accuracy_scores))
    np.save(auc_scores_save_path, np.array(auc_scores))
    if sample_indices is not None or attack_type == 'pgd':
        metadata_save_path = acc_scores_save_path.replace(
            '_acc.npy', '_metadata.npz'
        )
        metadata = {
            'attack': np.array(attack_type),
            'epsilons': np.asarray(eps_range, dtype=np.float64),
        }
        if attack_type == 'pgd':
            metadata.update({
                'max_iter': np.array(
                    max_iter if pgd_max_iter is None else pgd_max_iter,
                    dtype=np.int64,
                ),
                'num_random_init': np.array(
                    num_random_init
                    if pgd_num_random_init is None
                    else pgd_num_random_init,
                    dtype=np.int64,
                ),
                'eps_step_divisor': np.array(4.0, dtype=np.float64),
            })
        if sample_indices is not None:
            metadata.update({
                'sample_indices': np.asarray(sample_indices, dtype=np.int64),
                'test_subset_size': np.array(
                    len(sample_indices), dtype=np.int64
                ),
                'test_subset_seed': np.array(
                    transfer_seed, dtype=np.int64
                ),
            })
        np.savez(metadata_save_path, **metadata)
    if attack_type == 'cw':
        print("Saved acc, auc, and CW-L2 metrics.")
    else:
        print("Saved acc and auc scores.")
