#!/bin/bash

# List of objects
objects=(
#"bottle"
# "cable"
# "capsule"
# "carpet"
 "grid"
# "hazelnut"
# "leather"
# "metal_nut"
# "pill"
# "screw"
# "tile"
# "toothbrush"
#"transistor"
#"wood"
# "zipper"
)

# Directory paths
checkpoint_dir="/home/sfmt/PycharmProjects/vq-vae-2-pytorch/checkpoint"
data_dir="/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec"

# Iterate over the objects
for object in "${objects[@]}"
do
    # Run the extract_code.py command
    python extract_code.py --ckpt "$checkpoint_dir/${object}/${object}_vqvae_1000.pt" --name "lmdb/${object}" "$data_dir/$object/trainAugmented/" --gpu False
done
