import os
import shutil

categories = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor",
              "zipper", "carpet", "grid", "leather", "tile", "wood"]

source_base_path = os.path.join("", "mvtec_anomaly_detection")
destination_base_path = os.path.join("", "mvtec_anomaly_detection", "all", "train", "good")

# Ensure the destination base path exists
if not os.path.exists(destination_base_path):
    os.makedirs(destination_base_path)

def copy_with_new_names(source_folder, destination_folder):
    files = os.listdir(source_folder)

    for index, file in enumerate(files, 1):
        source_file_path = os.path.join(source_folder, file)
        new_filename = f"{category}_{file}"
        destination_file_path = os.path.join(destination_folder, new_filename)
        shutil.copy(source_file_path, destination_file_path)


for category in categories:
    source_folder = os.path.join(source_base_path, category, "train", "good")
    destination_folder = destination_base_path

    # Copy files from source to destination with new names
    copy_with_new_names(source_folder, destination_folder)
