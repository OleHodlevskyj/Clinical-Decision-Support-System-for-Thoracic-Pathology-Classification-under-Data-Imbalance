#створення нейромережі: готова архітектура з torchvision, у якій останній шар
#замінено так, щоб вона видавала 14 чисел — по одному на кожну патологію
import os
import torch.nn as nn
from torchvision import models


#повертає модель потрібної архітектури з num_classes виходами;
#pretrained=True — почати з ваг, навчених на ImageNet (для навчання),
#pretrained=False — порожня модель, у яку потім завантажують збережені ваги (для застосунку)
def build_model(num_classes=14, pretrained=True, arch=None):
    #архітектуру можна передати явно або змінною середовища ARCH, типово ResNet-50
    arch = arch or os.environ.get('ARCH', 'resnet50')
    if arch == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        #останній шар fc на ImageNet має 1000 виходів; ставимо новий на 14
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        #в EfficientNet останній шар лежить у classifier під номером 1
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise SystemExit(f'невідома архітектура ARCH={arch}: очікується resnet50 або efficientnet_b0')
