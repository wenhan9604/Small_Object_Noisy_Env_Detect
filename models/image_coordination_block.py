import torch
import torch.nn as nn


class ImageCoordinationBlock(nn.Module):
    def __init__(
            self,
            autoencoder,
            rcan,
            ffanet,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
            ):
        super().__init__()
        self.autoencoder = autoencoder
        self.rcan = rcan
        self.ffanet = ffanet
        self.groups = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            for _ in range(3)
        ])
        self.final_layers = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )
        self.attention = ChannelAttention(in_channels=out_channels)
    
    def forward(self, x):
        auto_encoder_output = self.autoencoder(x)
        ffanet_output = self.ffanet(x)
        rcan_output = self.rcan(x)

        # Each model passes through separate layers
        auto_encoder_output = self.groups[0](auto_encoder_output)
        ffanet_output = self.groups[1](ffanet_output)
        rcan_output = self.groups[2](rcan_output)

        # Concatenate the outputs from all three models
        x = torch.cat((auto_encoder_output, ffanet_output, rcan_output), dim=1)
        output = self.final_layers(x)
        output = self.attention(output)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels=64, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)
