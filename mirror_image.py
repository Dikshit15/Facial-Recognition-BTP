import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
    	if "flip" not in filename:
	        img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
	        	images.append(img)
    return images

def flip_image(images, folder):
    print(len(images))
    image_counter = 0
    for img in images:
        horizontal_image = cv2.flip(img, 1)
        cv2.imwrite(folder+"/image_flip_{}.png".format(image_counter), horizontal_image)
        image_counter+=1

folders = ["abhishek", "dikshit", "garvit", "jatin", "neelansh" , "saket"]
for folder in folders:
    flip_image(load_images_from_folder("dataset/" + folder), "dataset/" + folder)