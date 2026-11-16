# Train and evaluate CNN, ViT, and CapsNet models
# and conduct sensitivity, latent space, and 
# Grad-Cam experiments for Mnist models
# Aneja Lab | Yale School of Medicine
# Adapted from https://github.com/Yoega/BP-CapsNet/blob/main/train_and_eval.py
# and https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/train_and_eval_pytorch.py
# Author: Anand Srinivasan
# 05/29/25

import os
import argparse
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import trange
import PIL
import torch.optim as optim
import torch.utils.data as data
from loss import Criterion
import torch
import numpy as np
from copy import deepcopy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(665)

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, resnet50
from tensorboardX import SummaryWriter
from collections import OrderedDict

from models import capsnet as caps
from models import bpcapsnet as bpcaps
from models.resnets import ResNet18, ResNet50
from models.MedViT import MedViT_small
from scripts.experiments.sensitivity import *
from scripts.interpretability.grad_cam import *
from scripts.interpretability.latent_space import *
from utils import *

root = str(ROOT / 'data')


def save_checkpoint_atomic(state, path):
    """Write a checkpoint without exposing a partially written file."""
    temporary_path = path + '.tmp'
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def main(data_flag, num_epochs, gpu_ids, batch_size, download, model_flag,
        resize, as_rgb, model_path, run, experiment, attack_type, eps,
        test_subset_size=None, test_subset_seed=665,
        transfer_archive_path=None, transfer_epsilon=0.016):

    if data_flag != 'mnist':
       raise ValueError("Invalid dataset {}".format(data_flag))
    
    lr = 1e-4
    gamma = 0.96

    global n_channels
    n_channels = 1
    global n_classes
    n_classes = 10

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    if gpu_ids and torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to logical GPU 0.
        device = torch.device('cuda:0')
    else:
        if gpu_ids:
            print('CUDA is unavailable; falling back to CPU.')
        device = torch.device('cpu')

    print('==> Preparing data...')

    if '3d' in data_flag and resize:
        data_transform = transforms.Compose([
            MeanIntensityProjection(),
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    elif '3d' in data_flag:
        data_transform = transforms.Compose([
            MeanIntensityProjection(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    else:
        if resize:
            data_transform = transforms.Compose(
                [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])])
        else:
            data_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])])
    

    train_dataset = datasets.MNIST(root=root, train=True, transform=data_transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=data_transform, download=download)
    native_test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]
    )
    native_test_dataset = datasets.MNIST(
        root=root, train=False, transform=native_test_transform,
        download=download
    )

    test_indices = None
    if test_subset_size is not None:
        if not 1 <= test_subset_size <= len(test_dataset):
            raise ValueError(
                "--test_subset_size must be between 1 and {}".format(
                    len(test_dataset)
                )
            )
        generator = np.random.default_rng(test_subset_seed)
        test_indices = np.sort(
            generator.choice(
                len(test_dataset), size=test_subset_size, replace=False
            )
        )
        test_dataset = data.Subset(test_dataset, test_indices.tolist())
        native_test_dataset = data.Subset(
            native_test_dataset, test_indices.tolist()
        )
        print(
            "Using deterministic MNIST test subset: {} of {} images "
            "(seed={})".format(
                test_subset_size, len(native_test_dataset.dataset),
                test_subset_seed
            )
        )

    # A ViT sweep attacks the model's ordinary 224x224 inputs. Native inputs
    # are used only when explicitly saving a transferable archive inline.
    use_native_attack_inputs = resize and transfer_archive_path is not None
    attack_test_dataset = (
        native_test_dataset if use_native_attack_inputs else test_dataset
    )

    train_size = int(0.88 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_labels = np.array([label for _, label in train_dataset])
    val_labels = np.array([label for _, label in val_dataset])
    test_labels = np.array([label for _, label in test_dataset])

    train_evaluator = MNISTEvaluator('train', train_labels, root=root)
    val_evaluator = MNISTEvaluator('val', val_labels, root=root)
    test_evaluator = MNISTEvaluator('test', test_labels, root=root)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
    attack_test_loader = data.DataLoader(dataset=attack_test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)
    


    print('==> Building and training model...')

    if model_flag == 'resnet18':
        model = resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels,
                                                                                          num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels,
                                                                                          num_classes=n_classes)
    elif model_flag == 'caps':
        model = caps.CapsNet(n_channels, n_classes)
    elif model_flag == 'bpcaps':
        model = bpcaps.BPCapsNet(n_channels, n_classes)
    elif model_flag == 'vit':
        model = MedViT_small(n_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)

    criterion = Criterion(model_flag)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, model_flag, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, model_flag, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, model_flag, criterion, device, run)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_auc = -float('inf')
    best_epoch = 0
    best_model = deepcopy(model)

    checkpoint_path = None
    progress_checkpoint_path = None
    if num_epochs != 0:
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        resize_tag = "_224" if resize else ""
        checkpoint_path = os.path.join(
            save_dir,
            f"{model_flag}_{data_flag}{resize_tag}_best_model.pth",
        )
        progress_checkpoint_path = checkpoint_path.replace(
            "_best_model.pth", "_best_model_in_progress.pth"
        )

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, model_flag, criterion, optimizer, device, None)
        val_metrics = test(model, val_evaluator, val_loader, model_flag, criterion, device, run)
        print(
            'epoch: %d  train loss: %.5f  val auc: %.5f  val acc: %.5f'
            % (epoch + 1, train_loss, val_metrics[1], val_metrics[2])
        )

        scheduler.step()

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            state = {
                'net': best_model.state_dict(),
                'best_epoch': best_epoch,
                'best_val_auc': best_auc,
            }
            save_checkpoint_atomic(state, progress_checkpoint_path)
            print("Val improved!")
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)
            print('saved best checkpoint:', progress_checkpoint_path)

    if num_epochs != 0:
        if not os.path.isfile(progress_checkpoint_path):
            raise RuntimeError(
                "Training finished without producing a best checkpoint"
            )
        os.replace(progress_checkpoint_path, checkpoint_path)
        print("Saved final best model:", checkpoint_path)

    # Select checkpoints using validation AUC only. The held-out test set is
    # evaluated once, after training, using the best-validation model.
    if num_epochs != 0 or model_path is None:
        val_metrics = test(
            best_model, val_evaluator, val_loader,
            model_flag, criterion, device, run
        )
        test_metrics = test(
            best_model, test_evaluator, test_loader,
            model_flag, criterion, device, run
        )
        print(
            'best epoch: %d  val auc: %.5f  val acc: %.5f  '
            'test auc: %.5f  test acc: %.5f'
            % (
                best_epoch + 1,
                val_metrics[1], val_metrics[2],
                test_metrics[1], test_metrics[2],
            )
        )

    input_shape = attack_test_dataset[0][0].shape

    # ADV SENSITIVITY
    if experiment == 'sensitivity':
        generate_and_test_adversarial(best_model, input_shape,
                                   train_loader_at_eval, val_loader, attack_test_loader,
                                   optimizer, criterion, model_flag,
                                   train_evaluator, val_evaluator, test_evaluator,
                                   device, run, n_classes, data_flag, attack_type=attack_type,
                                   resize=resize, eps=eps,
                                   attack_resize=use_native_attack_inputs,
                                   transfer_archive_path=transfer_archive_path,
                                   transfer_epsilon=transfer_epsilon,
                                   transfer_seed=test_subset_seed,
                                   sample_indices=test_indices)
        
    # # ADV GRADIENTS
    # elif experiment == 'gradients':
    #     adversarial_gradient(best_model, input_shape,
    #                         train_loader_at_eval, val_loader, test_loader,
    #                         optimizer, criterion, model_flag,
    #                         train_evaluator, val_evaluator, test_evaluator,
    #                         device, run, n_classes, data_flag,
    #                         attack_type=attack_type, eps=eps)
    
    elif experiment == 'latent_space':
        x_test_adv, y_test_adv = generate_adversarial_examples_from_model_input(best_model, input_shape,
                                                    train_loader_at_eval, val_loader, test_loader,
                                                    optimizer, criterion, model_flag,
                                                    train_evaluator, val_evaluator, test_evaluator,
                                                    device, run, n_classes, data_flag,
                                                    attack_type=attack_type, eps=eps)
        adv_loader_custom = zip([torch.from_numpy(adv_batch) for adv_batch in x_test_adv], y_test_adv)
        plot_reduced_latent_space(best_model, test_loader, adv_loader_custom, model_flag, data_flag, device, dimensionality_reduction='tSNE', adversarial=True)
    
    elif experiment == 'gradcam':
        # GRAD CAM
        standard_grad_cams = grad_cam(best_model, test_loader, model_flag, data_flag, device, plot_cam=True)
        os.makedirs('gradcam_npy', exist_ok=True)
        np.save(f'gradcam_npy/{data_flag}_{model_flag}_{eps}_original.npy', standard_grad_cams)
        # GRAD CAM (adversarial)
        x_test_adv, y_test_adv = generate_adversarial_examples_from_model_input(best_model, input_shape,
                                                    train_loader_at_eval, val_loader, test_loader,
                                                    optimizer, criterion, model_flag,
                                                    train_evaluator, val_evaluator, test_evaluator,
                                                    device, run, n_classes, data_flag,
                                                    attack_type=attack_type, eps=eps)
        adv_loader_custom = zip([torch.from_numpy(adv_batch) for adv_batch in x_test_adv], y_test_adv)
        adv_flag = model_flag + '_adv_eps_{:.2f}'.format(eps)
        adv_grad_cams = grad_cam(best_model, adv_loader_custom, adv_flag, data_flag, device, plot_cam=True)
        np.save(f'gradcam_npy/{data_flag}_{model_flag}_{eps}_adversarial.npy', adv_grad_cams)

        # QUANTITATIVE GRAD CAM OVERLAP
        similarities = [gradcam_similarity(standard_cam, adv_cam, percent=20) for standard_cam, adv_cam in zip(standard_grad_cams, adv_grad_cams)]
        mean_similarity = np.mean(np.array(similarities))

        logging.basicConfig(
            filename='output.log',
            filemode='a',
            format='%(message)s',
            level=logging.DEBUG
        )

        log_msg = f"Model: {model_flag}, Dataset: {data_flag}, Attack type: {attack_type}, Eps: {eps} | GradCam similarity: {mean_similarity}"
        logging.info(log_msg)
    
    else:
        if experiment is None:
            print("No experiment chosen.")
        else:
            raise ValueError("Invalid experiment {}".format(experiment))



def train(model, train_loader, model_flag, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = torch.eye(n_classes).index_select(dim=0, index=targets).to(device)

        #print(targets.shape)

        reconstruction = None
        if model_flag[:3] == "res" or model_flag[:3] == "vit":
            outputs = model(inputs)
        else:
            outputs, reconstruction = model(inputs)

        # Compute loss & accuracy
        # --> traverse to tensor
        loss = criterion(inputs, targets, outputs, reconstruction)
        total_loss.append(loss.item())
        # writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, model_flag, criterion, device, run, save_folder=None):
    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # outputs and reconstruction of test inputs.to(device)
            inputs = inputs.to(device)
            reconstruction = None
            if model_flag[:3] == "res" or model_flag[:3] == "vit":
                outputs = model(inputs)
            else:
                outputs, reconstruction = model(inputs)
            
            targets = torch.eye(n_classes).index_select(dim=0, index=targets).to(device)
            loss = criterion(inputs, targets, outputs, reconstruction)
            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)
        print(test_loss, auc, acc)
        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline models and CapsNets of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='mnist',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=0,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='bpcaps',
                        help='choose backbone from resnet18, resnet50, capsnet, and bp-capsnet',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--experiment',
                        default=None,
                        help='to choose type of experiment from gradcam, latent space',
                        type=str)
    parser.add_argument('--attack_type',
                        default='pgd',
                        choices=['pgd', 'fgsm', 'bim', 'autoattack', 'cw'],
                        help='attack used for adversarial image generation',
                        type=str)
    parser.add_argument('--eps',
                        default=0.032,
                        help='epsilon used to generate adversarial images (ignored for CW)',
                        type=float)
    parser.add_argument('--test_subset_size',
                        default=None,
                        type=int,
                        help='deterministic number of MNIST test images to use')
    parser.add_argument('--test_subset_seed',
                        default=665,
                        type=int,
                        help='random seed used to select the MNIST test subset')
    parser.add_argument('--transfer_archive_path',
                        default=None,
                        type=str,
                        help='save sweep examples at --transfer_epsilon here')
    parser.add_argument('--transfer_epsilon',
                        default=0.016,
                        type=float)

    args = parser.parse_args()
    data_flag = args.data_flag
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    model_path = args.model_path
    run = args.run
    experiment = args.experiment
    attack_type = args.attack_type
    eps = args.eps
    test_subset_size = args.test_subset_size
    test_subset_seed = args.test_subset_seed
    transfer_archive_path = args.transfer_archive_path
    transfer_epsilon = args.transfer_epsilon

    main(data_flag, num_epochs, gpu_ids, batch_size, download, model_flag,
         resize, as_rgb, model_path, run, experiment, attack_type, eps,
         test_subset_size, test_subset_seed, transfer_archive_path,
         transfer_epsilon)
