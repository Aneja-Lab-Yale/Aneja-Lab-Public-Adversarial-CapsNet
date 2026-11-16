import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

class ToTensor3D:
    def __call__(self, sample):
        sample = sample.squeeze(0)
        sample = np.transpose(sample, (2, 0, 1))
        sample = torch.Tensor(sample)
        return sample

class MNISTEvaluator:
    def __init__(self, split, labels, root=None):
        """
        Evaluator for MNIST dataset.
        Args:
            split (str): 'train', 'val', or 'test' to indicate the split.
            root (str): Path to the dataset or data root.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")
        self.split = split
        self.labels = labels
        self.n_classes = 10  # MNIST has 10 classes (digits 0-9)

    def evaluate(self, y_score, save_folder=None, run=None):
        """
        Evaluates the predicted scores against the true labels.
        Args:
            y_score (np.array): The predicted probabilities (logits after softmax).
            save_folder (str): Directory to save evaluation logs (optional).
            run (str): Custom run name for logging purposes.
        Returns:
            dict: Dictionary containing 'auc' and 'accuracy'.
        """
        assert y_score.shape[0] == len(self.labels), "Number of predictions and labels must match."

        # One-hot encode the true labels for AUC calculation
        y_true_one_hot = np.eye(self.n_classes)[self.labels]

        # Calculate AUC for each class (one-vs-all)
        aucs = []
        for i in range(self.n_classes):
            try:
                auc = roc_auc_score(y_true_one_hot[:, i], y_score[:, i])
            except ValueError:
                auc = np.nan  # Handle case where AUC is undefined (e.g., if all values are same)
            aucs.append(auc)

        # Compute the mean AUC over classes and overall accuracy
        mean_auc = np.nanmean(aucs)
        acc = accuracy_score(self.labels, np.argmax(y_score, axis=1))

        metrics = (mean_auc, acc)
        print(f"AUC: {mean_auc:.4f}, ACC: {acc:.4f}")

        return metrics

def compute_iou(mask1, mask2, threshold=0.5):
    # Compute Intersection and Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    iou = intersection / union if union != 0 else 0
    return iou
