import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels: int = 1, nb_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=8, padding="same")
        self.bn1   = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding="same")
        self.bn2   = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding="same")
        self.bn3   = nn.BatchNorm1d(128)

        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, nb_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)  # (B, 128)
        logits = self.fc(x)          # (B, nb_classes)
        return logits