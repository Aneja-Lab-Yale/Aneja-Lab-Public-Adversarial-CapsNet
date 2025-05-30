#!/bin/bash

# Run sensitivity, Grad-Cam, latent space experiments
# for all Mnist models

script="experiments_mnist.py"

# Note: if data has not yet been downloaded, add --download flag
# upon first training a model on an undownloaded dataset

ARGS_LIST=(

    "--data_flag mnist --model_flag resnet18 --experiment sensitivity --model_path checkpoints/resnet18_mnist_best_model.pth"
    "--data_flag mnist --model_flag resnet50 --experiment sensitivity --model_path checkpoints/resnet50_mnist_best_model.pth"
    "--data_flag mnist --model_flag vit --experiment sensitivity --model_path checkpoints/vit_mnist_224_best_model.pth --resize"
    "--data_flag mnist --model_flag caps --experiment sensitivity --model_path checkpoints/caps_mnist_best_model.pth"
    "--data_flag mnist --model_flag bpcaps --experiment sensitivity --model_path checkpoints/bpcaps_mnist_best_model.pth"

    "--data_flag mnist --model_flag resnet18 --experiment latent_space --eps 0.256 --model_path checkpoints/resnet18_mnist_best_model.pth"
    "--data_flag mnist --model_flag resnet50 --experiment latent_space --eps 0.256 --model_path checkpoints/resnet50_mnist_best_model.pth"
    "--data_flag mnist --model_flag vit --experiment latent_space --eps 0.256 --model_path checkpoints/vit_mnist_224_best_model.pth --resize"
    "--data_flag mnist --model_flag caps --experiment latent_space --eps 0.256 --model_path checkpoints/caps_mnist_best_model.pth"
    "--data_flag mnist --model_flag bpcaps --experiment latent_space --eps 0.256 --model_path checkpoints/bpcaps_mnist_best_model.pth"

    "--data_flag mnist --model_flag resnet18 --experiment gradcam --eps 0.256 --model_path checkpoints/resnet18_mnist_best_model.pth"
    "--data_flag mnist --model_flag resnet50 --experiment gradcam --eps 0.256 --model_path checkpoints/resnet50_mnist_best_model.pth"
    "--data_flag mnist --model_flag vit --experiment gradcam --eps 0.256 --model_path checkpoints/vit_mnist_224_best_model.pth --resize"
    "--data_flag mnist --model_flag caps --experiment gradcam --eps 0.256 --model_path checkpoints/caps_mnist_best_model.pth"
    "--data_flag mnist --model_flag bpcaps --experiment gradcam --eps 0.256 --model_path checkpoints/bpcaps_mnist_best_model.pth"

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
