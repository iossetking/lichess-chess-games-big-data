"""Approach 3: the Maia / Leela-style residual tower as a drop-in policy net.

Maia is a family of human-like chess models trained on Lichess games by the
Computational Chess Lab; its published networks use a residual convolutional
tower with squeeze-excitation blocks (the ``lk`` architecture). This module
reproduces that architecture at a size that trains on a laptop, so the
secondary notebook can compare it against the from-scratch network using the
exact same training loop, dataset and legal-move masking.

Loading an official Maia checkpoint is a wrapper away: the checkpoint state
dict can be copied in with ``load_state_dict(..., strict=False)`` when its
channel widths match ``channels`` and ``num_blocks``.
"""

from __future__ import annotations

import torch
from torch import nn

from lichess_pipeline.features import NUM_MOVE_CLASSES

DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 4
SQUEEZE_RATIO = 4


class SqueezeExcitation(nn.Module):
    """Channel attention used by the Leela/Maia residual blocks."""

    def __init__(self, channels: int, ratio: int = SQUEEZE_RATIO) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = torch.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a squeeze-excitation skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return torch.relu(out + residual)


class MaiaResidualPolicyNet(nn.Module):
    """Residual policy tower with the same 4096-class move head as the baseline."""

    def __init__(
        self, channels: int = DEFAULT_CHANNELS, num_blocks: int = DEFAULT_BLOCKS
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(13, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(channels * 8 * 8, NUM_MOVE_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tower(self.stem(x))
        return self.head(torch.flatten(x, start_dim=1))
