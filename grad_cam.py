# Create Grad-Cam visualizations
# Aneja Lab | Yale School of Medicine
# Adapted from https://github.com/jacobgil/pytorch-grad-cam/tree/master
# Author: Anand Srinivasan
# 05/29/25

from typing import Callable, List, Optional, Tuple
import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision
import torch
import matplotlib.pyplot as plt
from utils import compute_iou

bloodmnist_label_dict = {0: 'neutrophil', 1: 'eosinophil', 2: 'basophil', 3: 'lymphocyte', 4: 'monocyte', 5: 'immature granulocyte', 6: 'eryhtroblast', 7: 'platelet'}
class CapsCam(GradCAM):
    def __init__(self, model, target_layers):
        super().__init__(model=model, target_layers=target_layers)
    
    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs, reconstruciton = self.activations_and_grads(input_tensor)
        self.outputs = outputs

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

def grad_cam(model, dataloader, model_flag, data_flag, device, plot_cam=False, threshold=None):

    output_dir = f"gradcam/{data_flag}/{model_flag}/"
    os.makedirs(output_dir, exist_ok=True)

    if model_flag[:3] == "res":
        target_layers = [model.layer4]
        cam_algorithm = GradCAM
    elif model_flag[:3] == "vit":
        target_layers = [model.features[18]]
        cam_algorithm = GradCAM
    else:
        target_layers = [model.conv]
        cam_algorithm = CapsCam
    
    grad_cams = []
    targets = None
    model.eval()
    with cam_algorithm(model=model,
                    target_layers=target_layers) as cam:
        cam.batch_size = 16
        batch_num = 0
        for input_tensor, labels in dataloader:

            # grid = torchvision.utils.make_grid(input_tensor)
            # grid_np = grid.numpy().transpose(1, 2, 0)
            # plt.imshow(grid_np)
            # plt.savefig(os.path.join(output_dir, 'baseline_imgs.png'))
            # plt.clf()

            rgb_img = input_tensor.permute(0, 2, 3, 1).cpu().numpy()
            grayscale_cams = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=False,
                                eigen_smooth=False)
            input_tensor = input_tensor.to(device)
            if model_flag[:3] == "res" or model_flag[:3] == "vit":
                preds = model(input_tensor)
            else:
                preds, reconstruciton = model(input_tensor)
            
            if plot_cam:
                for i in range(grayscale_cams.shape[0]):
                    grayscale_cam = grayscale_cams[i, :]
                    #print("Grayscale range {} - {}".format(grayscale_cam.min(), grayscale_cam.max()))

                    if data_flag == 'bloodmnist':
                        label = bloodmnist_label_dict[labels[i].item()]
                        pred = bloodmnist_label_dict[torch.argmax(preds[i]).item()]
                    elif data_flag == 'mnist':
                        label = labels[i]
                        pred = torch.argmax(preds[i]).item()                       
                    else:
                        label = labels[i].item()
                        pred = torch.argmax(preds[i]).item()
                    
                    if threshold is not None:
                        threshold_percentile = threshold
                        threshold = np.percentile(grayscale_cam, threshold_percentile)
                        mask = (grayscale_cam >= threshold).astype(np.float32)

                    # Apply colormap to the heatmap
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    heatmap_colored = np.float32(heatmap_colored / 255)

                    if threshold is not None:
                        heatmap_colored *= mask[:, :, np.newaxis]

                    recovered_img_weight = 1
                    recovered_img = recovered_img_weight * rgb_img[i] + (1 - recovered_img_weight) * heatmap_colored
                    img_weight = 0.5
                    mixed_image = img_weight * rgb_img[i] + (1 - img_weight) * heatmap_colored

                    np.save(f"{batch_num*16 + i}.npy", mixed_image)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                    axes[0].imshow(recovered_img)
                    axes[0].axis('off')
                    axes[0].set_title('Original Image: {}'.format(label))

                    axes[1].imshow(mixed_image)
                    axes[1].axis('off')
                    axes[1].set_title('{} Grad Cam: predicted {}'.format(model_flag, pred))

                    plt.savefig(os.path.join(output_dir, f"{batch_num*16 + i}.png"))
                    plt.clf()
                    plt.close()
            
                batch_num += 1
            
            grad_cams.append(grayscale_cams)
    
    grad_cams = np.concatenate(np.array(grad_cams), axis=0)
    return grad_cams


def gradcam_similarity(cam1, cam2, percent=15):
    # Flatten and sort pixel values
    flat1 = cam1.flatten()
    flat2 = cam2.flatten()
    
    # Get the threshold for the top `percent`% of values
    threshold1 = np.percentile(flat1, 100 - percent)
    threshold2 = np.percentile(flat2, 100 - percent)
    
    # Binarize the Grad-CAMs based on the threshold
    mask1 = cam1 >= threshold1
    mask2 = cam2 >= threshold2
    
    # Compute Intersection and Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    iou = intersection / union if union != 0 else 0
    return iou