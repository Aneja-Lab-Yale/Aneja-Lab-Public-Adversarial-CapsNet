#!/bin/bash

script="experiments_medmnist.py"

# Note: if data has not yet been downloaded, add --download flag
# upon first training a model on an undownloaded dataset

ARGS_LIST=(

    "--data_flag breastmnist --model_flag resnet18 --num_epochs 100"
    "--data_flag breastmnist --model_flag resnet50 --num_epochs 100"
    "--data_flag breastmnist --model_flag vit --num_epochs 100 --resize"
    "--data_flag breastmnist --model_flag caps --num_epochs 100"
    "--data_flag breastmnist --model_flag bpcaps --num_epochs 100"

    "--data_flag pneumoniamnist --model_flag resnet18 --num_epochs 100"
    "--data_flag pneumoniamnist --model_flag resnet50 --num_epochs 100"
    "--data_flag pneumoniamnist --model_flag vit --num_epochs 100 --resize"
    "--data_flag pneumoniamnist --model_flag caps --num_epochs 100"
    "--data_flag pneumoniamnist --model_flag bpcaps --num_epochs 100"

    "--data_flag nodulemnist3d --model_flag resnet18 --num_epochs 100"
    "--data_flag nodulemnist3d --model_flag resnet50 --num_epochs 100"
    "--data_flag nodulemnist3d --model_flag vit --num_epochs 100 --resize"
    "--data_flag nodulemnist3d --model_flag caps --num_epochs 100"
    "--data_flag nodulemnist3d --model_flag bpcaps --num_epochs 100"

    "--data_flag bloodmnist --model_flag resnet18 --num_epochs 100"
    "--data_flag bloodmnist --model_flag resnet50 --num_epochs 100"
    "--data_flag bloodmnist --model_flag vit --num_epochs 100 --resize"
    "--data_flag bloodmnist --model_flag caps --num_epochs 100"
    "--data_flag bloodmnist --model_flag bpcaps --num_epochs 100"

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
