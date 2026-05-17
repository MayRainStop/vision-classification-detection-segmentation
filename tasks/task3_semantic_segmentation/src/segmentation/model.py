from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 8, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.enc4 = ConvBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(widths[3], widths[3] * 2)

        self.up4 = nn.ConvTranspose2d(widths[3] * 2, widths[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(widths[3] * 2, widths[3])
        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(widths[2] * 2, widths[2])
        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(widths[1] * 2, widths[1])
        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(widths[0] * 2, widths[0])

        self.head = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))

        y4 = self.dec4(torch.cat([self.up4(x5), x4], dim=1))
        y3 = self.dec3(torch.cat([self.up3(y4), x3], dim=1))
        y2 = self.dec2(torch.cat([self.up2(y3), x2], dim=1))
        y1 = self.dec1(torch.cat([self.up1(y2), x1], dim=1))
        return self.head(y1)
