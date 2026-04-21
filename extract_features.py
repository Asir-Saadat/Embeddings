import torch
import numpy as np


def extract_features(model, loader):
    features = []
    labels = []

    with torch.no_grad():
        for imgs, targets in loader:
            out = model(imgs)                      
            out = out.squeeze(-1).squeeze(-1)  

            # print(out)

            features.append(out.numpy())

            # print(features)

            labels.append(targets.numpy())

            # print(fetures.shape())
            # print(features)

            # print(labels)

    features = np.concatenate(features) # (50000, 2048)
    labels = np.concatenate(labels)  # (50000,)
    np.save('features.npy', features)
    np.save('labels.npy', labels)
    return features, labels
