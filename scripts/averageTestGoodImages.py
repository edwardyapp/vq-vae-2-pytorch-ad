import os

def count_average_images(folder_paths):
    total_images = 0
    total_folders = 0

    for folder_path in folder_paths:
        good_folder_path = os.path.join(folder_path, "../test", "good")
        if os.path.exists(good_folder_path):
            num_images = len([name for name in os.listdir(good_folder_path) if os.path.isfile(os.path.join(good_folder_path, name))])
            total_images += num_images
            total_folders += 1

    if total_folders == 0:
        print("No 'good' folders found.")
    else:
        average_images = total_images / total_folders
        print(f"Average number of images in 'good' folders: {average_images}")

# List of folder paths to cycle through
folder_paths = [
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/bottle",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/cable",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/capsule",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/carpet",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/grid",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/hazelnut",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/leather",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/metal_nut",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/pill",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/screw",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/tile",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/toothbrush",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/transistor",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/wood",
    "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/zipper"
]

count_average_images(folder_paths)
