#!/bin/bash

# This script orchestrates a series of model training and testing operations.
# It includes running a baseline CNN model without pre-trained weights, and
# performing ablation studies on models with and without the jigsaw GAN components.

# Run the baseline CNN without any pre-trained model
echo "Starting baseline model..."
python main.py --mode baseline
echo "Baseline model complete."


# Run ablation tests on the  pretrained jigsaw GAN (9 piece)
echo "Starting ablation test for jigsaw GAN..."
./run_jigsaw_ablation.sh
echo "Jigsaw GAN ablation tests complete."

# Run ablation tests on the finetune the pretrained GAN (no jigsaw)
echo "Starting ablation test for non-jigsaw GAN..."
./run_non_jigsaw_ablation.sh
echo "Non-jigsaw GAN ablation tests complete."
