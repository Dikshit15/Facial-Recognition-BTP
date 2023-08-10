import os
import cv2
import numpy as np

# Input and output directories
input_folder = "dataset/sample"
output_folder = "dataset/augmentedDatasetDikshit/blurimages0.5"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Probability of adding blur to an image
blur_probability = 1  # Adjust as needed

# Range of kernel sizes for blur
min_kernel_size = 7
max_kernel_size = 25

# Loop through the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Load the image
        image = cv2.imread(input_path)
        
        # Add blur with a certain probability
        if np.random.rand() < blur_probability:
            kernel_size = np.random.randint(min_kernel_size, max_kernel_size + 1)
            kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1  # Ensure odd number
            blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            # Save the blurred image
            cv2.imwrite(output_path, blurred_image)
            
            print(f"Added blur to {filename} (kernel size: {kernel_size}) and saved as {output_path}")
        else:
            # If no blur, save the original image
            cv2.imwrite(output_path, image)
            print(f"Saved {filename} as {output_path}")
