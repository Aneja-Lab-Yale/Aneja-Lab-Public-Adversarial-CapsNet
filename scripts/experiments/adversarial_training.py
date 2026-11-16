"""Utilities for in-loop PGD adversarial training."""

import torch
import torch.nn.functional as functional


def classification_scores(model, inputs):
    """Return class scores from models with or without reconstruction output."""
    outputs = model(inputs)
    return outputs[0] if isinstance(outputs, tuple) else outputs


def pgd_linf(
    model,
    inputs,
    targets,
    epsilon,
    step_size,
    steps,
    random_start=True,
    clip_min=-1.0,
    clip_max=1.0,
):
    """Create label-aware, untargeted L-infinity PGD examples.

    The attack runs with the model in evaluation mode so batch-normalization
    statistics are not updated by intermediate PGD iterates. The model's
    original mode is restored before returning, and the returned tensor is
    detached from the attack graph.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if clip_min >= clip_max:
        raise ValueError("clip_min must be less than clip_max")

    clean = inputs.detach()
    labels = targets.detach().long().reshape(-1)
    was_training = model.training
    model.eval()

    if random_start:
        adversarial = clean + torch.empty_like(clean).uniform_(
            -epsilon, epsilon
        )
        adversarial = adversarial.clamp(clip_min, clip_max)
    else:
        adversarial = clean.clone()

    try:
        for _ in range(steps):
            adversarial.requires_grad_(True)
            scores = classification_scores(model, adversarial)
            loss = functional.cross_entropy(scores, labels)
            gradient = torch.autograd.grad(
                loss, adversarial, only_inputs=True
            )[0]

            adversarial = adversarial.detach() + step_size * gradient.sign()
            perturbation = (adversarial - clean).clamp(-epsilon, epsilon)
            adversarial = (clean + perturbation).clamp(clip_min, clip_max)
    finally:
        model.train(was_training)

    return adversarial.detach()
