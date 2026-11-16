# Train and evaluate CNN, ViT, and CapsNet models
# and conduct sensitivity, latent space, and 
# Grad-Cam experiments for MedMnist models
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
import logging


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(665)

import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from tensorboardX import SummaryWriter
from collections import OrderedDict

from models import capsnet as caps
from models import bpcapsnet as bpcaps
from models.resnets import ResNet18, ResNet50
from models.MedViT import MedViT_small
import medmnist
from medmnist import INFO
from scripts.experiments.sensitivity import *
from scripts.interpretability.grad_cam import *
from scripts.interpretability.latent_space import *
from utils import *
from scripts.experiments.adversarial_training import pgd_linf

root = str(ROOT / 'data')
os.makedirs(root, exist_ok=True)


def main(data_flag, num_epochs, gpu_ids, batch_size, download, model_flag,
        resize, as_rgb, model_path, run, experiment, attack_type, eps,
        adversarial_training=False, adv_eps=0.016, adv_steps=7,
        adv_step_size=None, adv_clean_weight=0.0, adv_random_start=True,
        output_tag=None, attack_epsilons=None, pgd_max_iter=None,
        pgd_num_random_init=None, output_subdir=None):
    lr = 1e-4
    gamma = 0.96

    info = INFO[data_flag]
    task = info['task']
    global n_channels
    n_channels = 3 if as_rgb else info['n_channels']
    global n_classes
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

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
            ToTensor3D(),
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    elif '3d' in data_flag:
        data_transform = transforms.Compose([
            ToTensor3D(),
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


    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, root=root)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, root=root)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, root=root)


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
    
    print(train_dataset[0][0].shape)

    print('==> Building and training model...')

    if '3d' in data_flag:
        n_channels = train_dataset[0][0].shape[0]

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

    train_evaluator = medmnist.Evaluator(data_flag, 'train', root=root)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', root=root)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', root=root)

    criterion = Criterion(model_flag)

    if adv_step_size is None:
        adv_step_size = adv_eps / 4
    if adversarial_training:
        if adv_eps <= 0:
            raise ValueError("--adv_eps must be positive")
        if adv_steps < 1:
            raise ValueError("--adv_steps must be at least 1")
        if adv_step_size <= 0:
            raise ValueError("--adv_step_size must be positive")
        if not 0.0 <= adv_clean_weight <= 1.0:
            raise ValueError("--adv_clean_weight must be between 0 and 1")
        print(
            "PGD adversarial training enabled: eps={}, steps={}, "
            "step_size={}, clean_weight={}, random_start={}".format(
                adv_eps,
                adv_steps,
                adv_step_size,
                adv_clean_weight,
                adv_random_start,
            )
        )

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

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        train_loss = train(
            model,
            train_loader,
            model_flag,
            criterion,
            optimizer,
            device,
            None,
            adversarial_training=adversarial_training,
            adv_eps=adv_eps,
            adv_steps=adv_steps,
            adv_step_size=adv_step_size,
            adv_clean_weight=adv_clean_weight,
            adv_random_start=adv_random_start,
        )

        train_metrics = test(model, train_evaluator, train_loader_at_eval, model_flag, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, model_flag, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, model_flag, criterion, device, run)


        scheduler.step()

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print("Val improved!")
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': best_model.state_dict(),
        'best_epoch': best_epoch,
        'adversarial_training': adversarial_training,
    }
    if adversarial_training:
        state['adversarial_training_config'] = {
            'attack': 'pgd',
            'epsilon': adv_eps,
            'steps': adv_steps,
            'step_size': adv_step_size,
            'clean_weight': adv_clean_weight,
            'random_start': adv_random_start,
            'clip_values': (-1.0, 1.0),
        }
    print("Training complete.")

    # Save best model if any trining occurred.
    if num_epochs != 0:
        save_dir = f"checkpoints/"
        os.makedirs(save_dir, exist_ok=True)
        if resize:
            checkpoint_name = f"{model_flag}_{data_flag}_224"
        else:
            checkpoint_name = f"{model_flag}_{data_flag}"
        if adversarial_training:
            checkpoint_name += "_pgd_at_eps_{:g}".format(adv_eps)
        path = os.path.join(save_dir, checkpoint_name + "_best_model.pth")
        torch.save(state, path)
        print("Saving best model to {}.".format(path))
    
    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, model_flag, criterion, device, run)
    val_metrics = test(best_model, val_evaluator, val_loader, model_flag, criterion, device, run)
    test_metrics = test(best_model, test_evaluator, test_loader, model_flag, criterion, device, run)

    input_shape = train_dataset[0][0].shape

    # ADV SENSITIVITY
    if experiment == 'sensitivity':
        generate_and_test_adversarial(best_model, input_shape,
                                   train_loader_at_eval, val_loader, test_loader,
                                   optimizer, criterion, model_flag,
                                   train_evaluator, val_evaluator, test_evaluator,
                                   device, run, n_classes, data_flag, attack_type=attack_type,
                                   resize=resize, eps=eps, output_tag=output_tag,
                                   attack_epsilons=attack_epsilons,
                                   pgd_max_iter=pgd_max_iter,
                                   pgd_num_random_init=pgd_num_random_init,
                                   output_subdir=output_subdir)
        
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
        standard_grad_cams = grad_cam(best_model, test_loader, model_flag, data_flag, device, plot_cam=False)
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
        adv_grad_cams = grad_cam(best_model, adv_loader_custom, adv_flag, data_flag, device, plot_cam=False)
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


def train(model, train_loader, model_flag, criterion, optimizer, device, writer,
          adversarial_training=False, adv_eps=0.016, adv_steps=7,
          adv_step_size=0.004, adv_clean_weight=0.0,
          adv_random_start=True):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        target_indices = targets.reshape(-1).long().to(device)
        targets = torch.eye(n_classes, device=device).index_select(
            dim=0, index=target_indices
        )

        if adversarial_training:
            adversarial_inputs = pgd_linf(
                model,
                inputs,
                target_indices,
                epsilon=adv_eps,
                step_size=adv_step_size,
                steps=adv_steps,
                random_start=adv_random_start,
                clip_min=-1.0,
                clip_max=1.0,
            )
        else:
            adversarial_inputs = inputs

        reconstruction = None
        if model_flag[:3] == "res" or model_flag[:3] == "vit":
            outputs = model(adversarial_inputs)
        else:
            outputs, reconstruction = model(adversarial_inputs)

        # Compute loss & accuracy
        loss = criterion(
            adversarial_inputs, targets, outputs, reconstruction
        )
        if adversarial_training and adv_clean_weight > 0:
            clean_reconstruction = None
            if model_flag[:3] == "res" or model_flag[:3] == "vit":
                clean_outputs = model(inputs)
            else:
                clean_outputs, clean_reconstruction = model(inputs)
            clean_loss = criterion(
                inputs, targets, clean_outputs, clean_reconstruction
            )
            loss = (
                (1.0 - adv_clean_weight) * loss
                + adv_clean_weight * clean_loss
            )
        total_loss.append(loss.item())
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

            targets = torch.squeeze(targets, dim=1)
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
                        default='breastmnist',
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
    parser.add_argument('--output_tag',
                        default=None,
                        help='optional label added to adversarial result filenames',
                        type=str)
    parser.add_argument('--output_subdir',
                        default=None,
                        help='optional subdirectory inside the attack output directory',
                        type=str)
    parser.add_argument('--attack_epsilons',
                        nargs='+',
                        default=None,
                        type=float,
                        help='explicit epsilon values to evaluate instead of the standard sweep')
    parser.add_argument('--pgd_max_iter',
                        default=None,
                        type=int,
                        help='override the default number of PGD iterations')
    parser.add_argument('--pgd_num_random_init',
                        default=None,
                        type=int,
                        help='override the default number of PGD random restarts')
    parser.add_argument('--adversarial_training',
                        action='store_true',
                        help='train on in-loop untargeted PGD examples')
    parser.add_argument('--adv_eps',
                        default=0.016,
                        type=float,
                        help='PGD L-infinity budget in normalized [-1, 1] space')
    parser.add_argument('--adv_steps',
                        default=7,
                        type=int,
                        help='number of PGD steps per training batch')
    parser.add_argument('--adv_step_size',
                        default=None,
                        type=float,
                        help='PGD step size (default: adv_eps / 4)')
    parser.add_argument('--adv_clean_weight',
                        default=0.0,
                        type=float,
                        help='weight of clean loss in [0, 1]; default trains only on adversarial inputs')
    parser.add_argument('--no_adv_random_start',
                        action='store_true',
                        help='start PGD at the clean input instead of randomly in the epsilon ball')

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

    main(
        data_flag, num_epochs, gpu_ids, batch_size, download, model_flag,
        resize, as_rgb, model_path, run, experiment, attack_type, eps,
        adversarial_training=args.adversarial_training,
        adv_eps=args.adv_eps,
        adv_steps=args.adv_steps,
        adv_step_size=args.adv_step_size,
        adv_clean_weight=args.adv_clean_weight,
        adv_random_start=not args.no_adv_random_start,
        output_tag=args.output_tag,
        attack_epsilons=args.attack_epsilons,
        pgd_max_iter=args.pgd_max_iter,
        pgd_num_random_init=args.pgd_num_random_init,
        output_subdir=args.output_subdir,
    )
