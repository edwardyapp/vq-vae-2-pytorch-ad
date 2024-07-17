#!/bin/bash

# Directory containing dataset folders
datasets_dir="/home/sfmt/PycharmProjects/anomalib/datasets/MVTec"

# Exclude this specific directory from the list
exclude_dir="/home/sfmt/PycharmProjects/anomalib/datasets/MVTec"

# Get a list of all dataset folders in the directory
datasets=($(find "$datasets_dir" -maxdepth 1 -type d -not -path "$exclude_dir"))

# Print out the list of dataset folders
#echo "List of dataset folders:"
#for dataset in "${datasets[@]}"; do
#    echo "$dataset"
#done

# Function to run the command for a given dataset on a specific GPU
run_command() {
    if [ -d "$1" ]; then
        CUDA_VISIBLE_DEVICES=$2 python ./train_vqvae.py "$1/train" --size 256 --epoch 10000 --early_stopping --patience 20
    fi
}

# Loop over each dataset and run the command
gpu_index=0
for dataset in "${datasets[@]}"; do
    run_command "$dataset" $gpu_index &
    ((gpu_index++))
    if [ $gpu_index -eq 4 ]; then
        gpu_index=0
        wait # Wait for the currently running scripts to complete
    fi
done

wait # Wait for any remaining scripts to complete
