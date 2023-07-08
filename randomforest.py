from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
data = pickle.load(open("output/embeddings.pickle", "rb"))

# Prepare the features and labels
X = data['embeddings']
y = data['labels']

# Instantiate and fit the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X, y)

# Print the predicted labels for the data points
print(rf.predict(X))
