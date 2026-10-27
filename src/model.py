import os
import torch.nn as nn
from torchvision import models


def build_model(num_classes=14, pretrained=True, arch=None):
    arch = arch or os.environ.get('ARCH', 'resnet50')
    if arch == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise SystemExit(f'невідома архітектура ARCH={arch}: очікується resnet50 або efficientnet_b0')
