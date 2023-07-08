# USAGE
# python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

# Instead of using the above command, use the below command:
# python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.joblib --le output/le.joblib

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from twilio.rest import Client
# import joblib
from joblib import load, dump

#Twilio setup
def send_message(name):
	account_sid ='AC00880b5290347b6ff68e3476f3522b5a'
	auth_token ='179e2446749c6876ae95b798662aadf6'
	client=Client(account_sid,auth_token)

	message = client.messages \
	            .create(
	                 body=name+" is waiting for you to meet at the door. Please recieve.",
	                 from_='+12015793833',
	                 to='+919079116089'
	             )

	print(message.sid)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
# recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = load(open(args["le"], "rb"))
# print(le)
# recognizer1 = pickle.loads(open("output/recognizer1.pickle", "rb").read())
recognizer = load(open("output/recognizer.joblib", "rb"))
recognizer1 = load(open("output/recognizer1.joblib", "rb"))
recognizer2 = load(open("output/recognizer2.joblib", "rb"))

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
frame_counter = 0
unknown_frame_counter = 0
svm_list = {}
random_forest_list = {}
knn_list = {}
# send_list = []
total_frames_passed = 0
# loop over frames from the video file stream
while total_frames_passed < 800:
	# grab the frame from the threaded video stream
	frame = vs.read()
	# if fl == 1:
	# 	break
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		
		# filter out weak detections
		if confidence > args["confidence"]:
			total_frames_passed += 1
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
			face = frame
			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			# faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
			# 	(96, 96), (0, 0, 0), swapRB=True, crop=False)
			# embedder.setInput(faceBlob)
			# vec = embedder.forward()
			# print(vec.shape)
			try:
				vec = face_recognition.face_encodings(face)[0]
				# print("face_recognition", vec.shape)
				vec = vec.reshape(1,-1)
			except IndexError:
				continue
			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			preds1 = recognizer1.predict_proba(vec)[0]
			preds2 = recognizer2.predict_proba(vec)[0]
			print("Preds is ", preds)
			print("Preds1 is ", preds1)
			print("Preds2 is ", preds2)
			# preds2 = preds2.reshape(-1,1)
			#print (preds)
			# for i in range(len(preds)):
			# 	preds[i]+=preds1[i]
			# 	preds[i]+=preds2[i]

			j = np.argmax(preds)
			j1 = np.argmax(preds1)
			j2 = np.argmax(preds2)
			proba = preds[j]
			name = le.classes_[j] 
			proba1 = preds[j1]
			name1 = le.classes_[j1]
			proba2 = preds[j2]
			name2 = le.classes_[j2]
			# draw the bounding box of the face along with the
			# associated probability
			if (proba > 0.8):
				# frame_counter += 1
				svm_list[name] = svm_list.get(name, 0) + 1
				# print("SVM name ", name)
			if(proba1 > 0.8):
				random_forest_list[name1] = random_forest_list.get(name1, 0) + 1
				# print("RandomForest name ", name1)
			if(proba2 > 0.8):
				knn_list[name2] = knn_list.get(name2, 0) + 1
				# print("kNN name ", name2)
				# knn_list[name2]+=1
				# if frame_counter%2000 == 0:
				# 	name_detected = max(name_list, key=name_list.count)
				# 	if name_detected not in send_list:
				# 		#send_message(name_detected)
				# 		send_list.append(name_detected)
				# 	print(name_detected)
				# 	name_list = []
				# 	break
				# else:
				# 	name_list.append(name)
				# text = "{}: {:.2f}%".format(name, proba * 100)
				# y = startY - 10 if startY - 10 > 10 else startY + 10
				# cv2.rectangle(frame, (startX, startY), (endX, endY),
				# 	(0, 0, 255), 2)
				# cv2.putText(frame, text, (startX, y),
				# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			# else:
			# 	unknown_frame_counter+=1
			# 	if unknown_frame_counter%2000 == 0:
					# send_message("An Unknown")


	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
print("SVM : ", svm_list)
print("RandomForest : ", random_forest_list)
print("kNN : ", knn_list)
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
