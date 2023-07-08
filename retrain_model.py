from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import argparse
import pickle
from sklearn.cluster import KMeans
from joblib import load, dump

recognizer = load(open("output/recognizer.pickle", "rb"))
# le = pickle.loads(open("output/le.pickle", "rb").read())

data = load(open("output/embeddings.pickle", "rb"))
le = LabelEncoder()
labels = le.fit_transform(data["names"])
new_recognizer = recognizer.fit(data['embeddings'], labels)
print("done retraining")
# f = open("output/recognizer.pickle", "wb")
# f.write(pickle.dumps(recognizer))
# f.close()
dump(new_recognizer, 'output/recognizer.joblib')