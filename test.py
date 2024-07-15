import os

# Directory containing the folders to search
base_directory = '/home/sfmt/PycharmProjects/thin-object-selection-grey/data/test_device1_images_merged'

# String to search for in file names
string_to_search = 'mask_object'

# Iterate through all subdirectories in the base directory
for root, dirs, files in os.walk(base_directory):

    # Initialize a counter for the total number of files
    total_files = 0

    for file in files:
        # Check if the file name contains the specified string
        if string_to_search in file:
            total_files += 1

    # Print the total number of files found
    print(f"Directory: {dirs}, Total files containing '{string_to_search}': {total_files}")

