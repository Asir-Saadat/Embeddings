import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(query_idx, features, dataset, top_k=5):

    #query_vec = features[query_idx]
    query_vec = features[query_idx].reshape(1, -1)

    # print(query_vec)

    sims = cosine_similarity(query_vec, features)[0]

    sorted_indices = np.flip(np.argsort(sims))
    sorted_indices = sorted_indices[sorted_indices != query_idx]
    top_indices = sorted_indices[:top_k]

    # print(top_indices)

    fig, axes = plt.subplots(1, top_k+1, figsize=(15, 3))
    axes[0].imshow(dataset[query_idx][0])
    axes[0].set_title('Query')
    axes[0].axis('off')

    for i, idx in enumerate(top_indices):
        axes[i+1].imshow(dataset[idx][0])
        axes[i+1].set_title(f'Match {i+1}')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig(f'retrieval_{query_idx}.png')
    plt.show()
   

    # print(top_indices)


