import torch
import torch.nn as nn
from torchvision import models

class DeepAgeNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepAgeNet, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)

        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x)