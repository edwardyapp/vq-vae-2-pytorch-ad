import os
from PIL import Image

# Specify the folder containing the PNG images
png_folder = "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/bottle/train/good/"

# Specify the output folder for the converted JPEG images
jpeg_folder = "/home/sfmt/PycharmProjects/vq-vae-2-pytorch/bottleJPG"
if not os.path.exists(jpeg_folder):
    os.makedirs(jpeg_folder)

# Iterate over the PNG images in the folder
for filename in os.listdir(png_folder):
    if filename.endswith(".png"):
        png_path = os.path.join(png_folder, filename)
        jpeg_path = os.path.join(jpeg_folder, os.path.splitext(filename)[0] + ".jpeg")

        # Open the PNG image and convert it to JPEG
        img = Image.open(png_path)
        img = img.convert("RGB")

        # Save the image as JPEG
        img.save(jpeg_path, "JPEG")

        print(f"Converted {filename} to JPEG.")

print("Conversion complete.")
