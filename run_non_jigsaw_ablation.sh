#!/bin/bash

# This script performs a series of fine-tuning experiments to investigate the impact of training data size on the performance
# of a GAN model pre-trained without using any jigsaw puzzle mechanism. The script methodically reduces the percentage
# of the training dataset used for fine-tuning from 100% to 10%, thereby studying how the lack of a jigsaw configuration
# influences model effectiveness across various data sizes.


# Starting value of percent
start=1.0

# Ending value of percent
end=0.1

# Step value for each iteration
step=0.1

# Loop from start to end, decrementing by step
for percent in $(seq $start -$step $end); do
    echo "Running model with percent = $percent"
    python main.py --mode ft --model_path ./pretrained_models/GAN_jigsaw0_64_200e/discriminator.pth --percent $percent
done
