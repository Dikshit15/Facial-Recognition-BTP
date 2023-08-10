import os
import cv2
import numpy as np

# Input and output directories
input_folder = "dataset/sample"
output_folder = "dataset/augmentedDatasetDikshit/occlusionprob0.5"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Probability of adding occlusions to an image
occlusion_probability = 1  # Adjust as needed

# Minimum and maximum size of occlusions
min_occlusion_size = 100
max_occlusion_size = 150

# Loop through the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Load the image
        image = cv2.imread(input_path)
        
        # Add occlusions with a certain probability
        if np.random.rand() < occlusion_probability:
            occluded_image = image.copy()
            
            # Generate random occlusion position and size
            x = np.random.randint(0, image.shape[1] - max_occlusion_size)
            y = np.random.randint(0, image.shape[0] - max_occlusion_size)
            occlusion_size = np.random.randint(min_occlusion_size, max_occlusion_size)
            
            # Apply occlusion (black rectangle) to the image
            occlusion = np.zeros((occlusion_size, occlusion_size, 3), dtype=np.uint8)
            occluded_image[y:y+occlusion_size, x:x+occlusion_size] = occlusion
            
            # Save the occluded image
            cv2.imwrite(output_path, occluded_image)
            
            print(f"Added occlusion to {filename} and saved as {output_path}")
        else:
            # If no occlusion, save the original image
            cv2.imwrite(output_path, image)
            
            print(f"Saved {filename} as {output_path}")
