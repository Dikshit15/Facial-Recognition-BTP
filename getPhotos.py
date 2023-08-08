import cv2
import os
import time

def capture_photos(folder_path, num_photos):
    print("Initializing the camera")
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Failed to open the webcam.")
        return

    photo_count = 0

    print("Camera started")

    while photo_count < num_photos:
        # Capture frame-by-frame
        print(f"Photo number {photo_count} printed")
        ret, frame = cap.read()

        if ret:
            # Display the resulting frame
            cv2.imshow('Webcam', frame)

            # Save the captured frame as an image file
            photo_path = os.path.join(folder_path, f"photo{photo_count + 1}.jpg")
            cv2.imwrite(photo_path, frame)

            photo_count += 1
            time.sleep(1)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Set the folder path to store the photos
folder_path = "photos"

# Set the number of photos to capture
num_photos = 100

# Call the function to capture photos
capture_photos(folder_path, num_photos)
