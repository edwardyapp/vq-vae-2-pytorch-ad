import os
import random
import shutil

def move_random_images(source_dir, dest_dir, fraction=0.5):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    file_list = os.listdir(source_dir)
    num_files_to_move = int(len(file_list) * fraction)

    random.seed(42)  # To ensure reproducibility
    files_to_move = random.sample(file_list, num_files_to_move)

    for file_name in files_to_move:
        source_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.move(source_file, dest_file)

if __name__ == "__main__":
    source_directory = "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/bottle/test/good/"
    destination_directory = "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/bottle/validation/good/"
    fraction_to_move = 0.5  # Half of the images will be moved

    move_random_images(source_directory, destination_directory, fraction=fraction_to_move)
