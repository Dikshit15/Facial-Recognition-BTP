import os
import cv2
import numpy as np

# Input and output directories
input_folder = "dataset/sample"
output_folder = "dataset/augmentedDatasetDikshit/artifactimages"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Parameters for noise and artifacts
# noise_probability = 1  # Probability of adding noise
artifact_probability = 1.0  # Probability of adding artifacts

# Loop through the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Load the image
        image = cv2.imread(input_path)

        noisy_image = image
        
        # Add noise with a certain probability
        # Increasing this value will result in more intense noise, while decreasing it will result in less intense noise.
        # if np.random.rand() < noise_probability:
        #     noisy_image = image + np.random.normal(0, 25, image.shape).astype(np.uint8)
        # else:
        #     noisy_image = image
        
        # Add artifacts with a certain probability
        if np.random.rand() < artifact_probability:
            num_artifacts = np.random.randint(6, 18)  # Number of artifacts to add
            for _ in range(num_artifacts):
                x = np.random.randint(0, image.shape[1])
                y = np.random.randint(0, image.shape[0])
                radius = np.random.randint(5, 30)
                color = tuple(np.random.randint(0, 256, size=(3,)).tolist())  # Convert to tuple
                cv2.circle(noisy_image, (x, y), radius, color, -1)
        
        # Clip pixel values to 0-255 range
        noisy_image = np.clip(noisy_image, 0, 255)
        
        # Save the noisy and artifact-ridden image
        cv2.imwrite(output_path, noisy_image)
        
        print(f"Generated noisy and artifact image from {filename} and saved as {output_path}")