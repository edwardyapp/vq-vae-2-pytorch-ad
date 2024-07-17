#!/bin/bash

# Your email address
recipient="edward_yapp@simtech.a-star.edu.sg"

# Define the subject and body of the email
subject="Script Complete - GPU 3"

# Your script's commands
objects=("transistor")
#objects=("transistor" "wood" "zipper")

for object in "${objects[@]}"; do
    body="Your train_vqvae script for $object has completed its execution."
    CUDA_VISIBLE_DEVICES=3 python ./train_vqvae.py /home/sfmt/Desktop/share01/MVTec/$object/trainAugmented --epoch 500 > checkpoint/$object/train_vqvae_$object.log 2>&1

    # Send an email once the script is complete
    if [ $? -eq 0 ]; then
        echo "$body" | mail -s "$subject" "$recipient"
    else
        echo "Script for $object encountered an error."
    fi
done
