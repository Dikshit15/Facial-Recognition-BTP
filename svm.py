from sklearn import svm
import pickle

# Load data
data = pickle.load(open("output/embeddings.pickle", "rb"))

# Prepare the features and labels
X = data['embeddings']
y = data['labels']

# Instantiate and fit the SVM classifier
svm_classifier = svm.SVC(random_state=0)
svm_classifier.fit(X, y)

# Print the predicted labels for the data points
print(svm_classifier.predict(X))
