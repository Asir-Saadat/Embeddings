import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_tsne(features, cluster_labels, n_samples=5000):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)

    embeddings_2d = tsne.fit_transform(features[:n_samples])
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
            c=cluster_labels[:n_samples], cmap='tab10', s=5, alpha=0.7)

    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE colored by K-means cluster')
    plt.savefig('tsne.png')
    plt.show()