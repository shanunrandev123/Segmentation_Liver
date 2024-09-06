import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.step(X)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.layer1 = DoubleConv(1, 32)
        self.layer2 = DoubleConv(32, 64)
        self.layer3 = DoubleConv(64, 128)
        self.layer4 = DoubleConv(128, 256)
        self.layer5 = DoubleConv(256 + 128, 128)
        self.layer6 = DoubleConv(128 + 64, 64)
        self.layer7 = DoubleConv(64 + 32, 32)
        self.layer8 = nn.Conv3d(32, 3, kernel_size=1)

        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        x4 = self.layer4(x3m)
        x5 = torch.nn.functional.interpolate(x4, scale_factor=2, mode='trilinear', align_corners=True)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)

        x6 = torch.nn.functional.interpolate(x5, scale_factor=2, mode='trilinear', align_corners=True)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)

        x7 = torch.nn.functional.interpolate(x6, scale_factor=2, mode='trilinear', align_corners=True)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)
        ret = self.layer8(x7)
        return ret
