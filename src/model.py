import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes=14, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
