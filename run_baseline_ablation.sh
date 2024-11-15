#!/bin/bash

# This script iteratively runs the baseline process with varying percentages of the training dataset.
# It starts with using 100% of the training data and decreases in steps of 10% down to 10%.


# Starting value of percent
start=1.0

# Ending value of percent
end=0.1

# Step value for each iteration
step=0.1

# Loop from start to end, decrementing by step
for percent in $(seq $start -$step $end); do
    echo "Running model with percent = $percent"
    python main.py --mode ft --percent $percent
done

