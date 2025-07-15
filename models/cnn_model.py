import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Глубокая сверточная нейронная сеть для классификации изображений ASL (29 классов).
    """
    def __init__(self, num_classes: int = 29) -> None:
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(3, 32),   # 64x64 -> 32x32
            self._conv_block(32, 64),  # 32x32 -> 16x16
            self._conv_block(64, 128), # 16x16 -> 8x8
            self._conv_block(128, 256),# 8x8 -> 4x4
            self._conv_block(256, 512) # 4x4 -> 2x2
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 512, 1, 1)
            nn.Flatten(),                  # (B, 512)
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)   # (B, num_classes)
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x