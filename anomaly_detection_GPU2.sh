#!/bin/bash

# Your email address
recipient="edward_yapp@simtech.a-star.edu.sg"

# Define the subject and body of the email
subject="Script Complete - GPU 0"
body="Your Bash script has completed its execution."

# List of object names
object_names=(
#  "bottle"
#  "cable"
#  "capsule"
#  "carpet"
#  "grid"
#  "hazelnut"
#  "leather"
#  "metal_nut"
  "pill"
  "screw"
  "tile"
  "toothbrush"
#  "transistor"
#  "wood"
#  "zipper"
)

# Iterate over each object name and run the commands
for object_name in "${object_names[@]}"; do
    # Run your command with the extracted filename
    CUDA_VISIBLE_DEVICES=2 python ./anomaly_detection.py --batch 167 \
      --top "all/n_res_channel_128_channel_128/all_pixelsnail_top_500.pt" \
      --bottom "all/default_pixelsnail_parameters/all_pixelsnail_bottom_500.pt" \
      --vqvae  "all/default_vqvae_parameters/all_vqvae_1000.pt" \
      --data_path "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec" \
      --class_name "$object_name" \
      --gpu \
      --resize 292 \
      --centerCrop 256 \
      --randomCrop 282 \
      --randomRotation 2
done

# Send an email once the script is complete
if [ $? -eq 0 ]; then
    echo "$body" | mail -s "$subject" "$recipient"
else
    echo "Script encountered an error."
fi
