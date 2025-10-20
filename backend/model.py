import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConv2d(nn.Module):
    """CoordConv layer with static coordinates for OpenVINO INT8 quantization"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, height=None, width=None):
        super(CoordConv2d, self).__init__()
        coord_channels = 2

        self.conv = nn.Conv2d(in_channels + coord_channels, out_channels,
                             kernel_size, stride, padding)

        # Pre-compute static coordinates for batch_size=1
        if height is not None and width is not None:
            x_coords = torch.linspace(-1.0, 1.0, width)
            y_coords = torch.linspace(-1.0, 1.0, height)

            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Shape: (1, 1, H, W) - batch_size=1, 1 channel, H height, W width
            xx = xx.unsqueeze(0).unsqueeze(0)
            yy = yy.unsqueeze(0).unsqueeze(0)

            # Register as buffers so they're moved to device with model
            self.register_buffer('xx_channel', xx, persistent=True)
            self.register_buffer('yy_channel', yy, persistent=True)
        else:
            raise ValueError("height and width must be provided for static coordinates")

    def forward(self, x):
        # Simply concatenate - coordinates will broadcast automatically from (1,1,H,W) to (B,1,H,W)
        x_with_coords = torch.cat([x, self.xx_channel, self.yy_channel], dim=1)
        return self.conv(x_with_coords)


class DoubleCoordConv(nn.Module):
    """Double convolution block with CoordConv"""

    def __init__(self, in_channels, out_channels, height, width):
        super(DoubleCoordConv, self).__init__()
        self.double_conv = nn.Sequential(
            CoordConv2d(in_channels, out_channels, kernel_size=3, padding=1, height=height, width=width),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, height, width):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleCoordConv(in_channels, out_channels, height, width)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv - NO DYNAMIC PADDING"""

    def __init__(self, in_channels, out_channels, height, width):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleCoordConv(in_channels, out_channels, height, width)

    def forward(self, x1, x2):
        # Completely static - no dynamic padding
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetCoordConv(nn.Module):
    """U-Net with CoordConv layers - fully static for OpenVINO INT8 quantization"""

    def __init__(self, n_channels, n_classes, input_size=512):
        super(UNetCoordConv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_size = input_size

        # Calculate exact dimensions for each level
        size_inc = input_size      # 512x512
        size_down1 = input_size // 2   # 256x256
        size_down2 = input_size // 4   # 128x128
        size_down3 = input_size // 8   # 64x64
        size_down4 = input_size // 16  # 32x32

        # Encoder
        self.inc = DoubleCoordConv(n_channels, 64, height=size_inc, width=size_inc)
        self.down1 = Down(64, 128, height=size_down1, width=size_down1)
        self.down2 = Down(128, 256, height=size_down2, width=size_down2)
        self.down3 = Down(256, 512, height=size_down3, width=size_down3)
        self.down4 = Down(512, 1024, height=size_down4, width=size_down4)

        # Decoder - dimensions match encoder levels going back up
        self.up1 = Up(1024, 512, height=size_down3, width=size_down3)  # 64x64
        self.up2 = Up(512, 256, height=size_down2, width=size_down2)   # 128x128
        self.up3 = Up(256, 128, height=size_down1, width=size_down1)   # 256x256
        self.up4 = Up(128, 64, height=size_inc, width=size_inc)        # 512x512

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits
