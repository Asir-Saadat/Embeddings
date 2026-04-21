import torch
import torchvision.models as models
import torch.nn as nn
import os
import numpy as np
from models.get_resnet import get_model
from data_loader import get_dataloader
from extract_features import extract_features
from clustering import cluster_features
from visualize import visualize_tsne
from torchvision.datasets import CIFAR10
from retrieval import retrieve
from contact_sheet import plot_contact_sheet
from purity import cluster_purity

def main():

    model = get_model()

    # print(model)

    loader, dataset = get_dataloader()

    if os.path.exists('features.npy'):
        features = np.load('features.npy')
        labels = np.load('labels.npy')
    else:
        features, labels = extract_features(model, loader)

    cluster_labels = cluster_features(features)

    # print(len(cluster_labels))

    # visualize_tsne(features, cluster_labels)

    # dataset_raw = CIFAR10(root='./data', train=True, download=False)
    # retrieve(0, features, dataset_raw, top_k=5)

    # plot_contact_sheet(cluster_labels, dataset_raw)

    purity = cluster_purity(cluster_labels, labels)
    print(f'Cluster purity: {purity:.4f}')



if __name__ == "__main__":
    main()