import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Decoder block: upsample then concatenate skip connection."""

    def __init__(self, in_ch: int, out_ch: int, trilinear: bool = True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose3d(
                in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dz = x2.size(2) - x1.size(2)
        dy = x2.size(3) - x1.size(3)
        dx = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2,
                        dy // 2, dy - dy // 2,
                        dz // 2, dz - dz // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric label-map segmentation.

    Args:
        in_channels:   Number of input modalities / channels.
        num_classes:   Number of segmentation classes (1 for binary).
        base_features: Feature channels at the first encoder level.
        trilinear:     Trilinear upsampling (False = transposed conv).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_features: int = 32,
        trilinear: bool = True,
    ):
        super().__init__()
        f = base_features
        factor = 2 if trilinear else 1

        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f,      f * 2)
        self.enc3 = Down(f * 2,  f * 4)
        self.enc4 = Down(f * 4,  f * 8)
        self.bottleneck = Down(f * 8, f * 16 // factor)

        self.dec4 = Up(f * 16, f * 8 // factor, trilinear)
        self.dec3 = Up(f * 8,  f * 4 // factor, trilinear)
        self.dec2 = Up(f * 4,  f * 2 // factor, trilinear)
        self.dec1 = Up(f * 2,  f,                trilinear)

        self.outc = nn.Conv3d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d = self.dec4(b,  e4)
        d = self.dec3(d,  e3)
        d = self.dec2(d,  e2)
        d = self.dec1(d,  e1)

        return self.outc(d)
