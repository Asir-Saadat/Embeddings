import numpy as np


def cluster_purity(cluster_labels, true_labels, n_clusters=10):
    total_correct = 0

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        true_labels_in_cluster = true_labels[cluster_indices]
        
        label_counts = [0] * 10
        for label in true_labels_in_cluster:
            label_counts[label] += 1
        
        most_common_count = 0
        for count in label_counts:
            if count > most_common_count:
                most_common_count = count

        total_correct += most_common_count
        

    purity = total_correct / len(cluster_labels)
    return purity