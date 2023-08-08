import cv2
import os

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory containing the images
image_directory = 'dataset/dikshit1'
output_directory = 'dataset/DikshitMaheshwari'

# Get a list of image filenames in the directory
print(os.listdir(image_directory))
image_filenames = [filename for filename in os.listdir(image_directory) if filename.endswith('.jpg')]

# Process up to 60 images
num_images_to_process = 100
for i, image_filename in enumerate(image_filenames):
    if i >= num_images_to_process:
        break
    
    image_path = os.path.join(image_directory, image_filename)
    image = cv2.imread(image_path)

    print("Image is ", image)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces and crop the facial region
    for j, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]  # Crop the face region
        save_path = os.path.join(output_directory, f'face_{i+1}_{j+1}.jpg')
        cv2.imwrite(save_path, face)
        print(f"Face {i+1}_{j+1} saved as {save_path}")
