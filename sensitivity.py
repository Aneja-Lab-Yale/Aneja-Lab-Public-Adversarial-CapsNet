# Evaluate models against adversarial attacks
# of varying strength
# Aneja Lab | Yale School of Medicine
# Adapted from https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-Adversarial-Imaging
# Authors: Anand Srinivasan, Marina Joel
# 05/29/25

from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
import argparse
from typing import Union
import os
import matplotlib.pyplot as plt


class CapsClassifier(PyTorchClassifier):
    def __init__(self, clip_values, model, optimizer, loss, input_shape, nb_classes, channels_first, device_type, is_not_caps=False):
        super().__init__(clip_values=clip_values, model=model, optimizer=optimizer, loss=loss, input_shape=input_shape, nb_classes=nb_classes, channels_first=channels_first, device_type=device_type)
        self.my_model = model
        self.is_not_caps = is_not_caps
        self.n_classes = nb_classes
    
    # Modify ART prediction function for multi output capsnet models
    def predict(  # pylint: disable=W0221
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Set model mode
        self._model.train(mode=training_mode)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Create dataloader
        x_tensor = torch.from_numpy(x_preprocessed)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        results_list = []
        for (x_batch,) in dataloader:
            # Move inputs to device
            x_batch = x_batch.to(self._device)

            # Run prediction
            with torch.no_grad():
                model_outputs = self._model(x_batch)
            output_full = model_outputs[-1]
            if (self.is_not_caps):
                output = output_full
            else:
                output, reconstruction = output_full
            output = output.detach().cpu().numpy().astype(np.float32)
            if len(output.shape) == 1:
                output = np.expand_dims(output, axis=1).astype(np.float32)

            results_list.append(output)

        results = np.vstack(results_list)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)


        return predictions

    # Modify ART loss_gradient function for multi output capsnet models
    def loss_gradient(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Union[np.ndarray, "torch.Tensor"],
        training_mode: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
                              Note on RNN-like models: Backpropagation through RNN modules in eval mode raises
                              RuntimeError due to cudnn issues and require training mode, i.e. RuntimeError: cudnn RNN
                              backward can only be called in training mode. Therefore, if the model is an RNN type we
                              always use training mode but freeze batch-norm and dropout layers if
                              `training_mode=False.`
        :return: Array of gradients of the same shape as `x`.
        """

        self._model.train(mode=training_mode)

        # Backpropagation through RNN modules in eval mode raises RuntimeError due to cudnn issues and require training
        # mode, i.e. RuntimeError: cudnn RNN backward can only be called in training mode. Therefore, if the model is
        # an RNN type we always use training mode but freeze batch-norm and dropout layers if training_mode=False.
        if self.is_rnn:
            self._model.train(mode=True)
            if not training_mode:
                logger.debug(
                    "Freezing batch-norm and dropout layers for gradient calculation in train mode with eval parameters"
                    "of batch-norm and dropout."
                )
                self.set_batchnorm(train=False)
                self.set_dropout(train=False)

        # Apply preprocessing
        if self.all_framework_preprocessing:
            if isinstance(x, torch.Tensor):
                x_grad = x.clone().detach().requires_grad_(True)
            else:
                x_grad = torch.tensor(x).to(self._device)
                x_grad.requires_grad = True
            if isinstance(y, torch.Tensor):
                y_grad = y.clone().detach()
            else:
                y_grad = torch.tensor(y).to(self._device)
            inputs_t, y_preprocessed = self._apply_preprocessing(x_grad, y=y_grad, fit=False, no_grad=False)
        elif isinstance(x, np.ndarray):
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)
            x_grad = torch.from_numpy(x_preprocessed).to(self._device)
            x_grad.requires_grad = True
            inputs_t = x_grad
        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)
        else:
            labels_t = y_preprocessed

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        if (self.is_not_caps):
            output = model_outputs[-1]
            reconstruction = None
            labels_t = labels_t.to(torch.float)
        else:
            output, reconstruction = model_outputs[-1]
            labels_t = labels_t.to(torch.float)
        
        loss = self._loss(inputs_t, labels_t, output, reconstruction)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        if self._use_amp:  # pragma: no cover
            from apex import amp  # pylint: disable=E0611

            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            loss.backward()

        if x_grad.grad is not None:
            if isinstance(x, torch.Tensor):
                grads = x_grad.grad
            else:
                grads = x_grad.grad.cpu().numpy().copy()
        else:
            raise ValueError("Gradient term in PyTorch model is `None`.")

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads




# ADVERSARIAL ATTACK CONFIGS (PGD)
max_iter = 30
num_random_init = 3

def test_classifier(classifier, evaluator, data_loader, device, n_classes, run, mnist=False):

    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # outputs and reconstruction of test inputs.to(device)
            #inputs = inputs.numpy()
            inputs = inputs
            outputs = classifier.predict(inputs)
            outputs = torch.from_numpy(outputs).to(device)
            if not mnist:
                targets = torch.squeeze(targets, dim=1) #comment out for mnist
            targets = torch.eye(n_classes).index_select(dim=0, index=targets).to(device)
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, None, run)

        return [auc, acc]

def generate_adversarial_examples(dataloader, attacker, device):
    adversarial_examples = []  # List to store adversarial examples
    labels = []                # List to store labels

    for inputs, targets in dataloader:
        # Move inputs and targets to the appropriate device
        #inputs, targets = inputs.to(device), targets.to(device)

        # Convert inputs to NumPy array (required by ART)
        inputs = inputs.cpu().numpy()

        # Generate adversarial examples
        adv_examples = attacker.generate(x=inputs)

        # Convert adversarial examples back to PyTorch tensors
        #adv_examples = torch.from_numpy(adv_examples_np).to(device)

        # Save the results
        adversarial_examples.append(adv_examples)
        labels.append(targets)

    return adversarial_examples, labels

def generate_adversarial_examples_from_model_input(
    model, input_shape,
    train_loader, val_loader, test_loader,
    optimizer, criterion, model_flag,
    train_evaluator, val_evaluator, test_evaluator,
    device, run, n_classes, data_flag,
    attack_type='fgsm', eps=0.01):
    
    if model_flag[:3] == "res" or model_flag[:3] == "vit":
        robust_classifier = CapsClassifier(clip_values=(0, 1), model=model, optimizer=optimizer, loss=criterion, input_shape=input_shape, nb_classes=n_classes, channels_first=True, device_type='gpu', is_not_caps=True)
    else:
        robust_classifier = CapsClassifier(clip_values=(0, 1), model=model, optimizer=optimizer, loss=criterion, input_shape=input_shape, nb_classes=n_classes, channels_first=True, device_type='gpu')
    
    # train_metrics = test_classifier(robust_classifier, train_evaluator, train_loader, device, n_classes, run)
    # val_metrics = test_classifier(robust_classifier, val_evaluator, val_loader, device, n_classes, run)
    # test_metrics = test_classifier(robust_classifier, test_evaluator, test_loader, device, n_classes, run)

    # print("Train acc {}, Val acc {}, Test acc {}".format(train_metrics[1], val_metrics[1], test_metrics[1]))
    # print("Train auc {}, Val auc {}, Test auc {}".format(train_metrics[0], val_metrics[0], test_metrics[0]))
    
    accuracy_scores = []
    auc_scores = []

    print("Creating attacks for eps: {}".format(eps))

    if attack_type == 'pgd':
        attacker = ProjectedGradientDescent(robust_classifier, eps=eps, eps_step=eps / 4, max_iter=max_iter,
                                        num_random_init=num_random_init)
    elif attack_type == 'fgsm':
        attacker = FastGradientMethod(robust_classifier, eps=eps)
    elif attack_type == 'bim':
        attacker = BasicIterativeMethod(robust_classifier, eps=eps, eps_step=eps/max_iter, max_iter=max_iter)
    else:
        print("Unrecognized attack type: {}".format(attack_type))
        exit(2)
        
    return generate_adversarial_examples(test_loader, attacker, device)

def generate_and_test_adversarial(
    model, input_shape,
    train_loader, val_loader, test_loader,
    optimizer, criterion, model_flag,
    train_evaluator, val_evaluator, test_evaluator,
    device, run, n_classes,
    data_flag, attack_type='pgd', resize=False):
    
    if model_flag[:3] == "res" or model_flag[:3] == "vit":
        robust_classifier = CapsClassifier(clip_values=(0, 1), model=model, optimizer=optimizer, loss=criterion, input_shape=input_shape, nb_classes=n_classes, channels_first=True, device_type='gpu', is_not_caps=True)
    else:
        robust_classifier = CapsClassifier(clip_values=(0, 1), model=model, optimizer=optimizer, loss=criterion, input_shape=input_shape, nb_classes=n_classes, channels_first=True, device_type='gpu')
    
    is_mnist = True if data_flag == 'mnist' else False
    train_metrics = test_classifier(robust_classifier, train_evaluator, train_loader, device, n_classes, run, is_mnist)
    val_metrics = test_classifier(robust_classifier, val_evaluator, val_loader, device, n_classes, run, is_mnist)
    test_metrics = test_classifier(robust_classifier, test_evaluator, test_loader, device, n_classes, run, is_mnist)

    print("Train acc {}, Val acc {}, Test acc {}".format(train_metrics[1], val_metrics[1], test_metrics[1]))
    print("Train auc {}, Val auc {}, Test auc {}".format(train_metrics[0], val_metrics[0], test_metrics[0]))

    eps_range = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]
    #eps_range_strong = [0.512, 1.024, 2.048, 4.096]

    # evaluate sensitivity scores of each image
    eps_scores = [1] * len(test_loader)
    
    accuracy_scores = []
    auc_scores = []

    save_dir = 'adv_output'
    os.makedirs(save_dir, exist_ok=True)

    if resize:
        acc_scores_save_path = f"adv_output/{data_flag}_{model_flag}_224_{attack_type}_acc_strong.npy"
        auc_scores_save_path = f"adv_output/{data_flag}_{model_flag}_224_{attack_type}_auc_strong.npy"
    else:
        acc_scores_save_path = f"adv_output/{data_flag}_{model_flag}_{attack_type}_acc_strong.npy"
        auc_scores_save_path = f"adv_output/{data_flag}_{model_flag}_{attack_type}_auc_strong.npy"

    for eps in eps_range:
        print("Creating attacks for eps: {}".format(eps))

        if attack_type == 'pgd':
            attacker = ProjectedGradientDescent(robust_classifier, eps=eps, eps_step=eps / 4, max_iter=max_iter,
                                            num_random_init=num_random_init)
        elif attack_type == 'fgsm':
            attacker = FastGradientMethod(robust_classifier, eps=eps)
        elif attack_type == 'bim':
            attacker = BasicIterativeMethod(robust_classifier, eps=eps, eps_step=eps/max_iter, max_iter=max_iter)
        else:
            print("Unrecognized attack type: {}".format(attack_type))
            exit(2)
        
        x_test_adv, y_test_adv = generate_adversarial_examples(test_loader, attacker, device)
        print("Generated {} adversarial examples for eps {}".format(attack_type, eps))

        adv_loader_custom = zip(x_test_adv, y_test_adv)
        adv_metrics = test_classifier(robust_classifier, test_evaluator, adv_loader_custom, device, n_classes, run, is_mnist)
        
        print("Eps {}: Acc - {} Auc - {}".format(eps, adv_metrics[1], adv_metrics[0]))

        auc_scores.append(adv_metrics[0])
        accuracy_scores.append(adv_metrics[1])
        #np.save(acc_scores_save_path, np.array(accuracy_scores, dtype='float'))
    
    np.save(acc_scores_save_path, np.array(accuracy_scores))
    np.save(auc_scores_save_path, np.array(auc_scores))
    print("Saved acc and auc scores.")

def compute_gradients(classifier, dataloader, device):
    batched_grads = []
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        grads = classifier.loss_gradient(x=inputs, y=targets)
        mean_gradient = torch.mean(grads, dim=(1, 2, 3))
        mean_absolute_gradient = torch.mean(torch.abs(grads), dim=(1, 2, 3))

        for i in range(inputs.shape[0]):
            print(f"Image {i + 1}:")
            print(f"  Mean Gradient: {mean_gradient[i]}")
            print(f"  Mean Absolute Gradient: {mean_absolute_gradient[i]}")
        print("End batch.")
        batched_grads.append(grads)
    return batched_grads

def visualize_gradients(grads, output_dir):
    batch_size = grads.shape[0]
    flat_grads = grads.reshape(batch_size, -1)
    print(flat_grads.shape)
    for i in range(flat_grads.shape[0]):
        plt.hist(flat_grads[i], bins=50, alpha=0.5, label="Image 1")
        plt.legend()
        plt.title(f"Gradient Histogram for Img {i}")
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{i}.png"))
        plt.close()

def adversarial_gradient(
    model, input_shape,
    train_loader, val_loader, test_loader,
    optimizer, criterion, model_flag,
    train_evaluator, val_evaluator, test_evaluator,
    device, run, n_classes,
    data_flag, attack_type='fgsm', eps=0.01):

    if model_flag[:3] == "res" or model_flag[:3] == "vit":
        robust_classifier = CapsClassifier(clip_values=(0, 1), model=model, optimizer=optimizer, loss=criterion, input_shape=input_shape, nb_classes=n_classes, channels_first=True, device_type='gpu', is_not_caps=True)
    else:
        robust_classifier = CapsClassifier(clip_values=(0, 1), model=model, optimizer=optimizer, loss=criterion, input_shape=input_shape, nb_classes=n_classes, channels_first=True, device_type='gpu')

    print("Creating attacks for eps: {}".format(eps))

    if attack_type == 'pgd':
        max_iter = 30
        num_random_init = 3
        attacker = ProjectedGradientDescent(robust_classifier, eps=eps, eps_step=eps / 4, max_iter=max_iter,
                                        num_random_init=num_random_init)
    elif attack_type == 'fgsm':
        attacker = FastGradientMethod(robust_classifier, eps=eps)
    elif attack_type == 'bim':
        attacker = BasicIterativeMethod(robust_classifier, eps=eps, eps_step=eps/max_iter, max_iter=max_iter)
    else:
        print("Unrecognized attack type: {}".format(attack_type))
        exit(2)

    x_test_adv, y_test_adv = generate_adversarial_examples(test_loader, attacker, device)
    print("Generated {} adversarial examples for eps {}".format(attack_type, eps))

    def one_hot(target_vector):
        target_vector = torch.squeeze(target_vector, dim=1)
        target_vector = torch.eye(n_classes, device=target_vector.device).index_select(dim=0, index=target_vector)
        return target_vector
    
    adv_loader_custom = zip([torch.from_numpy(adv_batch) for adv_batch in x_test_adv],
                            [one_hot(adv_batch_labels) for adv_batch_labels in y_test_adv])
    
    save_dir = f"gradients/{data_flag}/{model_flag}/"
    os.makedirs(save_dir, exist_ok=True)
    grads = compute_gradients(robust_classifier, adv_loader_custom, device)
    first_batch = grads[0].cpu().detach().numpy()
    visualize_gradients(first_batch, save_dir)
