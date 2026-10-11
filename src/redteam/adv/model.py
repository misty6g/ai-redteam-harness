"""A small CNN for MNIST (CPU-friendly)."""
from __future__ import annotations
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 100), nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
