import os

def count_defect_folders_and_average_images(folder_paths):
    total_defect_folders = 0
    total_images_in_defect_folders = 0

    for folder_path in folder_paths:
        test_folder_path = os.path.join(folder_path, "../test")
        if os.path.exists(test_folder_path):
            defect_folders = [name for name in os.listdir(test_folder_path) if os.path.isdir(os.path.join(test_folder_path, name)) and name != "good"]
            total_defect_folders += len(defect_folders)
            print("len(defect_folders): ", len(defect_folders))

            for defect_folder in defect_folders:
                defect_folder_path = os.path.join(test_folder_path, defect_folder)
                num_images = len([name for name in os.listdir(defect_folder_path) if os.path.isfile(os.path.join(defect_folder_path, name))])
                total_images_in_defect_folders += num_images

    if total_defect_folders == 0:
        print("No defect folders found.")
    else:
        average_images_in_defect_folders = total_images_in_defect_folders / total_defect_folders
        print(f"Number of defect folders across all categories: {total_defect_folders}")
        print(f"Average number of images in defect folders across all categories: {average_images_in_defect_folders}")

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

count_defect_folders_and_average_images(folder_paths)
