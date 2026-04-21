import torch.nn as nn
import torchvision.models as models


def get_model():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    return feature_extractor