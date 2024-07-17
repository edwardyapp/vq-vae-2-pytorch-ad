#!/bin/bash

# Your email address
recipient="edward_yapp@simtech.a-star.edu.sg"

# List of object names
object_names=(
  "bottle"
#  "cable"
#  "capsule"
#  "carpet"
#  "grid"
#  "hazelnut"
#  "leather"
#  "metal_nut"
#  "pill"
#  "screw"
#  "tile"
#  "toothbrush"
#  "transistor"
#  "wood"
#  "zipper"
)

# Iterate over each object name and run the commands
for object_name in "${object_names[@]}"; do
  folder="lmdb/$object_name"
  echo "Processing folder: $folder"
  subject="Script Complete - GPU 0 - $object_name"  # Include object_name in subject
  CUDA_VISIBLE_DEVICES=0 python ./train_pixelsnail.py $folder --epoch 500 --lr 1e-4 --hier top > checkpoint/${object_name}/train_pixelsnail_${object_name}_top.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python ./train_pixelsnail.py $folder --epoch 500 --lr 1e-4 --hier bottom > checkpoint/${object_name}/train_pixelsnail_${object_name}_bottom.log 2>&1
  echo "$body" | mail -s "$subject" "$recipient"
done

# Send an email once the script is complete
if [ $? -eq 0 ]; then
    echo "$body" | mail -s "$subject" "$recipient"
else
    echo "Script encountered an error."
fi