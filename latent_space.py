# Calculate 2D latent space embeddings 
# of original and adversarial test set images 
# Aneja Lab | Yale School of Medicine
# Author: Anand Srinivasan
# 05/29/25

import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os


def extract_latents(model, images, layer):
    model_layer = Model(inputs=model.input, outputs=layer.output)
    return model_layer.predict(images)

def plot_reduced_latent_space(model, dataloader, dataloader_adversarial, model_flag, data_flag, device, dimensionality_reduction='tSNE', adversarial=False):

    activations = {}
    if model_flag[:3] == "res":
        def hook_fn(module, input, output):
            activations["latent_space"] = output
        hook = model.layer4.register_forward_hook(hook_fn)
    elif model_flag == "vit":
        def hook_fn(module, input, output):
            activations["latent_space"] = output
        hook = model.avgpool.register_forward_hook(hook_fn)  # Register hook at avgpool
    elif model_flag == "caps":
        def hook_fn(module, input, output):
            activations["latent_space"] = output
        hook = model.digit_caps.register_forward_hook(hook_fn)
    else:
        def hook_fn(module, input, output):
            activations["latent_space"] = output[0]  # Capture `out` from (out, v1, v2)
        hook = model.routing_module.register_forward_hook(hook_fn)
    
    model.eval()
    original_count = 0
    all_latent = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:

            inputs = inputs.to(device)
            reconstruction = None
            if model_flag[:3] == "res" or model_flag[:3] == "vit":
                outputs = model(inputs)
            else:
                outputs, reconstruction = model(inputs)

            latent_space = activations["latent_space"]

            if 'caps' in model_flag:
                probabilities = outputs
            else:
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            latent_flat = latent_space.view(latent_space.shape[0], -1)
            latent_flat = latent_flat.cpu().numpy()
            all_latent.append(latent_flat)

            if data_flag != 'mnist':
                targets = torch.squeeze(targets, dim=1)
            targets = targets.cpu().numpy()
            all_targets.append(targets)
            original_count += targets.shape[0]
        
        for inputs, targets in dataloader_adversarial:

            inputs = inputs.to(device)
            reconstruction = None
            if model_flag[:3] == "res" or model_flag[:3] == "vit":
                outputs = model(inputs)
            else:
                outputs, reconstruction = model(inputs)

            latent_space = activations["latent_space"]

            if 'caps' in model_flag:
                adv_probabilities = outputs
            else:
                adv_probabilities = torch.nn.functional.softmax(outputs, dim=1)

            latent_flat = latent_space.view(latent_space.shape[0], -1)
            latent_flat = latent_flat.cpu().numpy()
            all_latent.append(latent_flat)

            if data_flag != 'mnist':
                targets = torch.squeeze(targets, dim=1)
            targets = targets.cpu().numpy()
            all_targets.append(targets)
    
    combined_latent = np.concatenate(all_latent, axis=0)
    combined_targets = np.concatenate(all_targets, axis=0)

    mean = np.mean(combined_latent, axis=0)
    std = np.std(combined_latent, axis=0)
    normalized_latent = (combined_latent - mean) / std

    if dimensionality_reduction == 'tSNE':
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        combined_reduced_latent = tsne.fit_transform(combined_latent)
    elif dimensionality_reduction == 'pca':
        pca = PCA(n_components=2)
        combined_reduced_latent = pca.fit_transform(combined_latent)

    output_adv_file = f"{dimensionality_reduction}_adversarial.png"
    output_og_file = f"{dimensionality_reduction}.png"
    output_dir = output_path = f"latent_space/{data_flag}/{model_flag}/"
    os.makedirs(output_dir, exist_ok=True)
    output_adv_path = os.path.join(output_dir, output_adv_file)
    output_og_path = os.path.join(output_dir, output_og_file)

    og_latent = combined_reduced_latent[:original_count]
    adv_latent = combined_reduced_latent[original_count:]
    og_targets = combined_targets[:original_count]
    adv_targets = combined_targets[original_count:]

    output_npz_file = os.path.join(output_dir, f"{dimensionality_reduction}_latent_data.npz")
    np.savez(output_npz_file, og_latent=og_latent, adv_latent=adv_latent, og_targets=og_targets, adv_targets=adv_targets)

    
