import dlib
import cv2
import os
import numpy as np
from imutils import face_utils

# Path to shape predictor model
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"  # You need to provide this file

# Input and output directories
input_folder = "dataset/sample"
output_folder = "dataset/augmentedDatasetDikshit/agedimages"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the shape predictor model
predictor = dlib.shape_predictor(shape_predictor_path)
detector = dlib.get_frontal_face_detector()  # Initialize the face detector

# Define aging parameters
age_factor = 20  # Adjust as needed

# Loop through the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        
        # Load the image
        image = cv2.imread(input_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect facial landmarks
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Define aging transformation (example: scaling along Y-axis)
            for i in range(17, 68):
                shape[i][1] = int(shape[i][1] * age_factor)
            
            # Apply affine transformation to warp the facial features
            M = cv2.estimateAffinePartial2D(shape, shape)[0]
            warped = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            
            # Save the aged image
            aged_output_path = os.path.join(output_folder, f"aged_{filename}")
            cv2.imwrite(aged_output_path, warped)
            
            print(f"Aged {filename} and saved as {aged_output_path}")
