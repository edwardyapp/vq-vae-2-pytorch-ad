import os
import cv2

def get_image_resolutions(folder_path):
    resolutions = set()
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath)
            if img is not None:
                resolutions.add((img.shape[1], img.shape[0]))  # (width, height)
    return resolutions

folder_path = "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/all/train/good"
unique_resolutions = get_image_resolutions(folder_path)
print("Unique Resolutions:")
for resolution in unique_resolutions:
    print(f"{resolution[0]} x {resolution[1]}")
