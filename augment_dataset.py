import os
import cv2
import numpy as np

# Input and output directories
input_folder = "dataset/sample"
output_folder = "dataset/augmentedDatasetDikshit/darknessFactor0.4"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Darkening factor
darkening_factor = 0.4  # Adjust as needed

# Loop through the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Load the image
        image = cv2.imread(input_path)
        
        # Darken the image by reducing pixel values
        darkened_image = np.clip(image * darkening_factor, 0, 255).astype(np.uint8)
        
        # Save the darkened image
        cv2.imwrite(output_path, darkened_image)
        
        print(f"Darkened {filename} and saved as {output_path}")