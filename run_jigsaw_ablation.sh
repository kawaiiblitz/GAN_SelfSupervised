#!/bin/bash

# This script performs a series of fine-tuning experiments to analyze the effect of training data size on model performance
# using a pre-trained GAN model that incorporates a jigsaw puzzle mechanism during its training phase.
# The script decreases the percentage of the training dataset utilized for fine-tuning from 100% to 10%,
# reducing the dataset size by 10% in each run to systematically explore the impact of data size on the effectiveness of the jigsaw-enhanced model.


# Starting value of percent
start=1.0

# Ending value of percent
end=0.1

# Step value for each iteration
step=0.1

# Loop from start to end, decrementing by step
for percent in $(seq $start -$step $end); do
    echo "Running model with percent = $percent"
    python main.py --mode ft --model_path ./pretrained_models/GAN_jigsaw9_64_200e/discriminator.pth --percent $percent
done

