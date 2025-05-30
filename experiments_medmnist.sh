#!/bin/bash

# Run sensitivity, Grad-Cam, latent space experiments
# for all MedMnist models

script="experiments_medmnist.py"

# Note: if data has not yet been downloaded, add --download flag
# upon first training a model on an undownloaded dataset

ARGS_LIST=(

    # Breastmnist Experiments
    "--data_flag breastmnist --model_flag resnet18 --experiment sensitivity --model_path checkpoints/resnet18_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag resnet50 --experiment sensitivity --model_path checkpoints/resnet50_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag vit --experiment sensitivity --model_path checkpoints/vit_breastmnist_224_best_model.pth --resize"
    "--data_flag breastmnist --model_flag caps --experiment sensitivity --model_path checkpoints/caps_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag bpcaps --experiment sensitivity --model_path checkpoints/bpcaps_breastmnist_best_model.pth"

    "--data_flag breastmnist --model_flag resnet18 --experiment latent_space --model_path checkpoints/resnet18_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag resnet50 --experiment latent_space --model_path checkpoints/resnet50_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag vit --experiment latent_space --model_path checkpoints/vit_breastmnist_224_best_model.pth --resize"
    "--data_flag breastmnist --model_flag caps --experiment latent_space --model_path checkpoints/caps_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag bpcaps --experiment latent_space --model_path checkpoints/bpcaps_breastmnist_best_model.pth"

    "--data_flag breastmnist --model_flag resnet18 --experiment gradcam --model_path checkpoints/resnet18_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag resnet50 --experiment gradcam --model_path checkpoints/resnet50_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag vit --experiment gradcam --model_path checkpoints/vit_breastmnist_224_best_model.pth --resize"
    "--data_flag breastmnist --model_flag caps --experiment gradcam --model_path checkpoints/caps_breastmnist_best_model.pth"
    "--data_flag breastmnist --model_flag bpcaps --experiment gradcam --model_path checkpoints/bpcaps_breastmnist_best_model.pth"

    # Pneumoniamnist Experiments
    "--data_flag pneumoniamnist --model_flag resnet18 --experiment sensitivity --model_path checkpoints/resnet18_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag resnet50 --experiment sensitivity --model_path checkpoints/resnet50_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag vit --experiment sensitivity --model_path checkpoints/vit_pneumoniamnist_224_best_model.pth --resize"
    "--data_flag pneumoniamnist --model_flag caps --experiment sensitivity --model_path checkpoints/caps_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag bpcaps --experiment sensitivity --model_path checkpoints/bpcaps_pneumoniamnist_best_model.pth"

    "--data_flag pneumoniamnist --model_flag resnet18 --experiment latent_space --model_path checkpoints/resnet18_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag resnet50 --experiment latent_space --model_path checkpoints/resnet50_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag vit --experiment latent_space --model_path checkpoints/vit_pneumoniamnist_224_best_model.pth --resize"
    "--data_flag pneumoniamnist --model_flag caps --experiment latent_space --model_path checkpoints/caps_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag bpcaps --experiment latent_space --model_path checkpoints/bpcaps_pneumoniamnist_best_model.pth"

    "--data_flag pneumoniamnist --model_flag resnet18 --experiment gradcam --model_path checkpoints/resnet18_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag resnet50 --experiment gradcam --model_path checkpoints/resnet50_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag vit --experiment gradcam --model_path checkpoints/vit_pneumoniamnist_224_best_model.pth --resize"
    "--data_flag pneumoniamnist --model_flag caps --experiment gradcam --model_path checkpoints/caps_pneumoniamnist_best_model.pth"
    "--data_flag pneumoniamnist --model_flag bpcaps --experiment gradcam --model_path checkpoints/bpcaps_pneumoniamnist_best_model.pth"

    # Nodulemnist3d Experiments
    "--data_flag nodulemnist3d --model_flag resnet18 --experiment sensitivity --model_path checkpoints/resnet18_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag resnet50 --experiment sensitivity --model_path checkpoints/resnet50_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag vit --experiment sensitivity --model_path checkpoints/vit_nodulemnist3d_224_best_model.pth --resize"
    "--data_flag nodulemnist3d --model_flag caps --experiment sensitivity --model_path checkpoints/caps_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag bpcaps --experiment sensitivity --model_path checkpoints/bpcaps_nodulemnist3d_best_model.pth"

    "--data_flag nodulemnist3d --model_flag resnet18 --experiment latent_space --model_path checkpoints/resnet18_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag resnet50 --experiment latent_space --model_path checkpoints/resnet50_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag vit --experiment latent_space --model_path checkpoints/vit_nodulemnist3d_224_best_model.pth --resize"
    "--data_flag nodulemnist3d --model_flag caps --experiment latent_space --model_path checkpoints/caps_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag bpcaps --experiment latent_space --model_path checkpoints/bpcaps_nodulemnist3d_best_model.pth"

    "--data_flag nodulemnist3d --model_flag resnet18 --experiment gradcam --model_path checkpoints/resnet18_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag resnet50 --experiment gradcam --model_path checkpoints/resnet50_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag vit --experiment gradcam --model_path checkpoints/vit_nodulemnist3d_224_best_model.pth --resize"
    "--data_flag nodulemnist3d --model_flag caps --experiment gradcam --model_path checkpoints/caps_nodulemnist3d_best_model.pth"
    "--data_flag nodulemnist3d --model_flag bpcaps --experiment gradcam --model_path checkpoints/bpcaps_nodulemnist3d_best_model.pth"

    # Bloodmnist Experiments
    "--data_flag bloodmnist --model_flag resnet18 --experiment sensitivity --model_path checkpoints/resnet18_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag resnet50 --experiment sensitivity --model_path checkpoints/resnet50_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag vit --experiment sensitivity --model_path checkpoints/vit_bloodmnist_224_best_model.pth --resize"
    "--data_flag bloodmnist --model_flag caps --experiment sensitivity --model_path checkpoints/caps_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag bpcaps --experiment sensitivity --model_path checkpoints/bpcaps_bloodmnist_best_model.pth"

    "--data_flag bloodmnist --model_flag resnet18 --experiment latent_space --model_path checkpoints/resnet18_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag resnet50 --experiment latent_space --model_path checkpoints/resnet50_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag vit --experiment latent_space --model_path checkpoints/vit_bloodmnist_224_best_model.pth --resize"
    "--data_flag bloodmnist --model_flag caps --experiment latent_space --model_path checkpoints/caps_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag bpcaps --experiment latent_space --model_path checkpoints/bpcaps_bloodmnist_best_model.pth"

    "--data_flag bloodmnist --model_flag resnet18 --experiment gradcam --model_path checkpoints/resnet18_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag resnet50 --experiment gradcam --model_path checkpoints/resnet50_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag vit --experiment gradcam --model_path checkpoints/vit_bloodmnist_224_best_model.pth --resize"
    "--data_flag bloodmnist --model_flag caps --experiment gradcam --model_path checkpoints/caps_bloodmnist_best_model.pth"
    "--data_flag bloodmnist --model_flag bpcaps --experiment gradcam --model_path checkpoints/bpcaps_bloodmnist_best_model.pth"

)

for ARGS in "${ARGS_LIST[@]}"; do
    echo "Running: python $script $ARGS"
    # Note: replace with path to your env's python
    ~/miniconda3/envs/pytorch_env/bin/python3 "$script" $ARGS
    if [ $? -ne 0 ]; then
        echo "Error occurred while running the script with arguments: $ARGS"
        exit 1
    fi
done

echo "All executions completed successfully."
