from sklearn.cluster import KMeans
import numpy as np


def cluster_features(features, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    # print(cluster_labels)
    return cluster_labels
