import torch
import torchvision.models as models
import torch.nn as nn
import os
import numpy as np
from models.get_resnet import get_model
from data.data_loader import get_dataloader
from extract_features import extract_features
from clustering import cluster_features
from visualize import visualize_tsne

def main():

    # Getting the model
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

    visualize_tsne(features, cluster_labels)




if __name__ == "__main__":
    main()