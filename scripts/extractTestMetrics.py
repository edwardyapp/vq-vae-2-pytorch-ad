import re
import csv
import pandas as pd

# Define the categories and their corresponding order
category_order = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper", "carpet", "grid", "wood", "leather", "tile"]

# Define the metrics to extract
metrics = ["image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score"]

# Create a dictionary to store metric DataFrames
metric_dataframes = {}

# Iterate over the metrics
for metric in metrics:
    # Create a dictionary to store the metric values
    metric_values = {}

    # Iterate over the categories and extract the metric values
    for category in category_order:
        # Read the contents of the file
        file_path = f"/home/sfmt/PycharmProjects/vq-vae-2-pytorch/mvtec_result/{category}/{category}_testMetrics.txt"
        with open(file_path, "r") as file:
            file_contents = file.read()

            # Use regular expressions to find the relevant information
            pattern = rf"{metric}\s*\|\s*([0-9.]+)"
            match = re.search(pattern, file_contents)
            if match:
                # Extract the metric value
                metric_value = match.group(1)
                metric_values[category] = float(metric_value)
            else:
                print(f"No {metric} found for category: {category}")

    # Convert metric values to a DataFrame
    metric_df = pd.DataFrame(metric_values, index=[metric])
    # Add the DataFrame to the dictionary
    metric_dataframes[metric] = metric_df

# Write the metric DataFrames to separate tabs in an Excel file
excel_writer = pd.ExcelWriter("../metrics.xlsx")
for metric, df in metric_dataframes.items():
    df.to_excel(excel_writer, sheet_name=metric)
excel_writer.close()

