import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class DeepAgeNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepAgeNet, self).__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x)