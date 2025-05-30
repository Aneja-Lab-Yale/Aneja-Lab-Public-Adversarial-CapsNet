# On the Adversarial Robustness of Medical Capsule Networks

## Overview
We train two deep neural networks (ResNet-18, ResNet-50),
a vision transformer based model (MedViT), and two capsules networks
(DR-Caps, BP-Caps) on MNIST and four MedMNIST medical classification
datasets. We test all models against adversarial attacks and find 
capsule networks remain the most robust to these perturbations. Latent-space
and Grad-Cam visualizations reveal that capsule networks maintain more
stable feature spaces and attention maps after the attacks.

## Contents:
Use these links to easily navigate through this readme:

1. [Installation and Set Up](#getting-started)
2. [File Structure](#file-structure)
3. [Public Data](#data)
4. [Training](#training)
5. [Adversarial Sensitivity](#sensitivity)
6. [Latent Space and Grad-Cam](#interpretability)
7. [References](#references)



## Installation and Set Up:
### Getting Started:
1) First clone this repository:

```bash
git clone https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-Adversarial-CapsNet
```

2) Navigate to the cloned repository and run the following command to create the appropriate environment with the appropriate software packages:

```bash
cd Aneja-Lab-Public-Adversarial-CapsNet
conda env create -f adversarial_env.yml
conda activate adversarial_env
```

3) Create data directory, where MedMnist and Mnist data will be downloaded:
```bash
mkdir data
```

### File Structure:
The cloned repository will have the following file structure. Descriptions of major folders and files are below:

```bash
.
├── data                       # folder to store all MedMnist and MNIST data
├── models                     # contains model definitions for ResNet, MedViT, and CapsNet models
    ├── MedViT.py                       # MedViT model definition
    ├── bpcapsnet.py                    # BP-CapsNet model definition
    ├── capsnet.py                      # Standard (DR) CapsNet model definition
    ├── resnets.py                      # ResNet-18 and ResNet-50 model definitions
├── adversarial_env.yml        # YAML configuration file specifying virtual env details
├── experiments_medmnist.py    # python code for training all MedMNIST models and running all adversarial sensitivity, latent space, and Grad-Cam experiments
├── experiments_medmnist.sh    # executable to run all MedMNIST adversarial sensitivity, latent space, and Grad-Cam experiments
├── experiments_mnist.py       # python code for training all MNIST models and running all adversarial sensitivity, latent space, and Grad-Cam experiments
├── experiments_mnist.sh       # executable to run all MNIST adversarial sensitivity, latent space, and Grad-Cam experiments
├── grad_cam.py                # python implementation of Grad-Cam attention map visualization
├── latent_space.py            # python implementation of feature sapce -> latent space calculations
├── loss.py                    # python code containing loss functions
├── sensitivity.py             # python implementation of adversarial attacks and adversarial sensitivity experiments
├── train_medmnist.sh          # executable to train all MedMNIST models
├── train_mnist.sh             # executable to train all MNIST models
```
## Data:

All data used in this study is publicly available. MNIST is downloaded from the torchvision.datasets module.
Descriptions of all MedMNIST datasets can be found here. For downloading all datasets for our experiments,
simply include the --download flag in the command line arguments (see more detailed information below(#training).

## Training:

## Adversarial Sensitivity

## Interpretability
### Latent Space

### Grad-Cam

## References:
[Hossain, S., Hossain, S., Sritharan, D., Fu, D., Nene, A., Hossain, J., Chadha, S., Kim, I., Lin, M., Aboian, M. and Aneja, S., 2025. Deep Learning Identified Extra-Prostatic Extension and Seminal Vesicle Invasion as an MRI Biomarker for Prostate Cancer Outcomes. medRxiv, pp.2024-12.](https://www.medrxiv.org/content/10.1101/2024.12.31.24319822v1)

[Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.](https://www.nature.com/articles/s41592-020-01008-z)
 
