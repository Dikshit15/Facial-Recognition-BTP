# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
# Use the below one using joblib
# python train_model.py --embeddings output/embeddings.joblib --le output/le.joblib

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import argparse
import pickle
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from joblib import load, dump

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = load('output/embeddings.joblib')
print("Dikshit printing " , data)

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model SVC...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data['embeddings'], labels)

print("[INFO] training model RandomForestClassifier...")
recognizer1 = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=0)
recognizer1.fit(data['embeddings'], labels)

print("[INFO] training model kNearestNeighbors...")
recognizer2 = KNeighborsClassifier(n_neighbors=7)
recognizer2.fit(data['embeddings'], labels)

# recognizer1.fit(data['embeddings'], labels)



# recognizers = [ 
# 				MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,128,128), random_state=1),
# 				SVC(C=1.0, kernel="linear", probability=True, gamma='scale'),
# 				KNeighborsClassifier(n_neighbors = 10),
# 				SVC(C=1.0, kernel="rbf", probability=True,gamma='scale'),
# 				SVC(C=1.0, kernel="poly", probability=True,gamma='scale')
# 			  ]
# for i in recognizers:
# 	i.fit(data['embeddings'], labels)


# write the actual face recognition model to disk
# f = open(args["recognizer"], "wb")
# f.write(dump(recognizer, 'output/recognizer.joblib'))
# f.close()

dump(recognizer, 'output/recognizer.joblib')
dump(recognizer1, 'output/recognizer1.joblib')
dump(recognizer2, 'output/recognizer2.joblib')

# f1= open("output/recognizer1.pickle","wb")
# f1.write(pickle.dumps(recognizer1))
# f1.close()


# f2= open("output/recognizers.pickle","wb")
# f2.write(pickle.dumps(recognizers))
# f2.close()


# write the label encoder to disk
# f = open(args["le"], "wb")
dump(le, 'output/le.joblib')
# f.write(dump(le, 'output/le.pickle'))
# f.close()
