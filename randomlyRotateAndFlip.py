import os
import cv2
import random

# Define a function to randomly rotate an image within a specified angle range
def random_rotate(image, min_angle, max_angle):
    angle = random.uniform(min_angle, max_angle)
    height, width, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# Folder path containing the images
# folder_path = '/home/sfmt/Desktop/share01/MVTec/'
folder_path = '/home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/'

# List of categories
# categories = ['bottle', 'hazelnut', 'metal_nut', 'screw', 'cable', 'capsule', 'pill', 'toothbrush', 'transistor', 'zipper',
#               'carpet', 'grid', 'tile', 'leather', 'wood']
categories = ['grid']

# Iterate over categories
for category in categories:
    # Folder path for the current category
    category_folder_path = os.path.join(folder_path, f'{category}/train/good')

    # Output folder path for the current category
    category_output_folder = os.path.join(folder_path, f'{category}/trainAugmented/good')

    # Create the output folder if it does not exist
    if not os.path.exists(category_output_folder):
        os.makedirs(category_output_folder)

    # Iterate over files in the folder
    for filename in os.listdir(category_folder_path):
        # Check if the file has an image extension
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Full path to the image file
            file_path = os.path.join(category_folder_path, filename)

            # Load the image
            image = cv2.imread(file_path)

            if category in ['cable', 'capsule', 'pill', 'toothbrush', 'transistor', 'zipper']:
                # Randomly rotate each image between -10 and 10 degrees and do this 10 times
                for i in range(10):
                    rotated_image = random_rotate(image, -10, 10)

                    # Generate a new filename with rotation information
                    new_filename = os.path.splitext(filename)[0] + f'_rot_{i}' + os.path.splitext(filename)[1]

                    # Save the rotated image
                    output_path = os.path.join(category_output_folder, new_filename)
                    cv2.imwrite(output_path, rotated_image)
            else:
                # Get the image dimensions
                height, width, _ = image.shape

                # Generate all possible transformations
                transformations = []
                for rotation_angle in [0, 90, 180, 270]:
                    for flip_horizontal in [True, False]:
                        for flip_vertical in [True, False]:
                            transformations.append((rotation_angle, flip_horizontal, flip_vertical))

                # Randomly choose 8 transformations without replacement
                chosen_transformations = random.sample(transformations, k=8)

                # Apply the chosen transformations
                for transformation in chosen_transformations:
                    rotation_angle, flip_horizontal, flip_vertical = transformation

                    # Make a copy of the original image
                    transformed_image = image.copy()

                    # Rotate the image
                    if rotation_angle != 0:
                        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)
                        transformed_image = cv2.warpAffine(transformed_image, rotation_matrix, (width, height))

                    # Flip the image
                    if flip_horizontal:
                        transformed_image = cv2.flip(transformed_image, 1)  # Flip horizontally
                    if flip_vertical:
                        transformed_image = cv2.flip(transformed_image, 0)  # Flip vertically

                    # if category in ['carpet', 'grid', 'tile', 'leather', 'wood']:
                    #     # Random crop to 256x256
                    #     if height > 256 and width > 256:
                    #         y = random.randint(0, height - 256)
                    #         x = random.randint(0, width - 256)
                    #         transformed_image = transformed_image[y:y + 256, x:x + 256]

                    # Generate a new filename with rotation and flip information
                    new_filename = os.path.splitext(filename)[0] + f'_rot{rotation_angle}_flipH{flip_horizontal}_flipV{flip_vertical}' + os.path.splitext(filename)[1]

                    # Save the rotated and flipped image
                    output_path = os.path.join(category_output_folder, new_filename)
                    cv2.imwrite(output_path, transformed_image)

# Print a message after processing all images
print("Rotation and flipping complete!")