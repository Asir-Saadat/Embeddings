import numpy as np
import matplotlib.pyplot as plt

def plot_contact_sheet(cluster_labels,dataset,n_clusters=10,n_samples=5):

    fig, axes = plt.subplots(n_clusters, n_samples, figsize=(20, 20))

    for cluster_id in range(n_clusters):
        cluster_indices= np.where(cluster_labels==cluster_id)[0]
        sampled= np.random.choice(cluster_indices, n_samples, replace=False)

        for j, idx in enumerate(sampled):
            axes[cluster_id][j].imshow(dataset[idx][0])
            axes[cluster_id][j].axis('off')

        axes[cluster_id][0].set_title(f'C{cluster_id}', fontsize=8)



    plt.tight_layout()
    plt.savefig('contact_sheet.png')
    plt.show()