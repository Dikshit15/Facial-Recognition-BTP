import cv2
import os

# Input and output directories
input_folder = "dataset/sample"
output_folder = "dataset/augmentedDatasetDikshit/rotatedimages"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Angles for rotation (in degrees)
rotation_angles = [0, 30, 60, 90, 120, 150, 180]  # Adjust as needed

# Loop through the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_prefix = os.path.splitext(filename)[0]
        
        # Load the image
        image = cv2.imread(input_path)
        
        # Rotate and save images for each angle
        for angle in rotation_angles:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotate clockwise by 90 degrees
            
            # Save the rotated image
            rotated_output_path = os.path.join(output_folder, f"{output_prefix}_rotated_{angle}.jpg")
            cv2.imwrite(rotated_output_path, rotated_image)
            
            print(f"Rotated {filename} by {angle} degrees and saved as {rotated_output_path}")
