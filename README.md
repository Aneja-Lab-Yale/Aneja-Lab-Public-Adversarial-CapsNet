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
5. [Adversarial Sensitivity](#adversarial-sensitivity)
6. [Interpretability](#latent-space)
7. [References](#references)



## Installation and Set Up:
### Getting Started:
1) First clone this repository:

```bash
git clone https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-Adversarial-CapsNet
```

2) Navigate to the cloned repository and run the following commands to create the appropriate environment with the appropriate software packages:

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
Descriptions of all MedMNIST datasets can be found [here](https://medmnist.com/). For downloading all datasets for our experiments,
simply include the --download flag in the command line arguments (see more detailed information [below](#training)).

## Training:
### MNIST
To train all models (ResNets-18, ResNet-50, MedViT, DR-Caps, BP-Caps)
on MNIST simply run:

```bash
./train_mnist.sh
```


### MedMNIST
To train all models (ResNets-18, ResNet-50, MedViT, DR-Caps, BP-Caps)
on the four selected MedMNIST datasets (BreastMNIST, PneumoniaMNIST,
NoduleMNIST3D, BloodMNIST) simply run:

```bash
./train_medmnist.sh
```

Notes:
- To only run a certain set of experiments, comment out lines in ARG_LIST
  corresponding to experiments you wish to exclude.
- If a certain dataset has not been downloaded, include the **--download** flag in the command line arguments
  within the appropriate shell script for the first model that uses this dataset.
- The MedViT model uses the **--resize** flag to resize input images from size (28,28) -> (224, 224)
  following the original paper.
- Training for MNIST experiments is currently set to 100 epochs. This may take a while (a few days)
to run. Good results can be achieved with 5-10 epochs of training. You can change
the training time in train_mnist.sh with the **--num_epochs** command line argument.
- Training for MedMNIST datasets is relatively quick and we have found 100 epochs (current default) to result
  in a reasonable training time.

After training, all models should be stored in a folder called _checkpoints_.

## Adversarial Attacks:

Once all models have been trainined, to run all adversarial sensitivty, latent space, and grad-cam
experiments for MNIST models simply run:

```bash
./experiments_mnist.sh
```

To run all adversarial sensitivty, latent space, and grad-cam experiments for MNIST models simply run:

```bash
./experiments_medmnist.sh
```

Notes:
- To only run a certain set of experiments, comment out lines in ARG_LIST
  corresponding to experiments you wish to exclude.
- The **--experiment** argument controls the type of experiment being run.
  It can be set to {sensitivity, latent_space, grad_cam}.

### Adversarial Sensitivity:
In adversarial sensitivity experiments, models are tested against adversarial attacks of
varying strength (eps = {0.001, 0.002, 0.004, ... 0.256}). To change the attack type, use
the **--attack_type** argument to choose between {pgd=default, fgsm}. Results will be stored
as npy files in a folder called _adv_output_.

### Latent Space:
In latent space experiments, a model's latent space embeddings for all test set examples are visualized
before and after an adversarial attack. A tSNE dimensionality reduction is used to reduce the latent embeddings
into 2D space. To change the attack type, use the **--attack_type** argument to choose between {pgd=default, fgsm}.
To change the strneght of the attack, use the **--eps** arguments to set the values of epsilon (default=0.032).

### Grad-Cam:
In Grad-Cam experiments, attention maps for all test set examples are visualized
before and after an adversarial attack. To change the attack type, use the **--attack_type** argument to choose
between {pgd=default, fgsm}. To change the strneght of the attack, use the **--eps** arguments to set the values
of epsilon (default=0.032).

## References:
[Lei, Y., Wu, Z., Li, Z., Yang, Y. and Liang, Z., 2023. BP-CapsNet: An image-based Deep Learning method for medical diagnosis. *Applied Soft Computing*, 146, 110683.](https://www.sciencedirect.com/science/article/pii/S1568494623007019)

[Manzari, O. N., Ahmadabadi, H., Kashiani, H., Shokouhi, S. B., and Ayatollahi, A., 2023. MedViT: A robust vision transformer for generalized medical image classification. *Computers in Biology and Medicine*, 157, 106791.](https://doi.org/10.1016/j.compbiomed.2023.106791)

[Torres, J. and Gildenblat, J., 2021. pytorch-grad-cam: Gradient-weighted Class Activation Mapping (Grad-CAM) in PyTorch. *GitHub repository*.](https://github.com/jacobgil/pytorch-grad-cam)

[Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H. and Ni, B., 2023. MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*.](https://www.nature.com/articles/s41597-022-01721-8)
