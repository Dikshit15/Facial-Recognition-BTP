from sklearn.cluster import KMeans
import pickle
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

kmeans = KMeans(n_clusters = 7, random_state = 0).fit(data['embeddings'])
print(kmeans.labels_)