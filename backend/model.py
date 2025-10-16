import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConv2d(nn.Module):
    """CoordConv layer that adds coordinate channels to input"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CoordConv2d, self).__init__()
        coord_channels = 2
        
        # Add 2 channels for x,y coordinates (and optionally 1 for radius)
        self.conv = nn.Conv2d(in_channels + coord_channels, out_channels, 
                             kernel_size, stride, padding)
        
    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Create coordinate channels
        xx_channel = torch.arange(width, dtype=x.dtype, device=x.device).repeat(1, 1, height, 1)
        yy_channel = torch.arange(height, dtype=x.dtype, device=x.device).repeat(1, 1, width, 1).transpose(2, 3)
        
        # Normalize coordinates to [-1, 1]
        xx_channel = xx_channel.float() / (width - 1)
        yy_channel = yy_channel.float() / (height - 1)
        xx_channel = (xx_channel - 0.5) * 2
        yy_channel = (yy_channel - 0.5) * 2
        
        # Expand to batch size
        xx_channel = xx_channel.expand(batch_size, -1, -1, -1)
        yy_channel = yy_channel.expand(batch_size, -1, -1, -1)
        
        # Concatenate coordinate channels
        coord_channels = [xx_channel, yy_channel]
        
        x_with_coords = torch.cat([x] + coord_channels, dim=1)
        return self.conv(x_with_coords)


class DoubleCoordConv(nn.Module):
    """Double convolution block with CoordConv"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleCoordConv, self).__init__()
        self.double_conv = nn.Sequential(
            CoordConv2d(in_channels, out_channels, kernel_size=3, padding=1),
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
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleCoordConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleCoordConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetCoordConv(nn.Module):
    """U-Net with CoordConv layers"""
    
    def __init__(self, n_channels, n_classes):
        super(UNetCoordConv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleCoordConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
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
