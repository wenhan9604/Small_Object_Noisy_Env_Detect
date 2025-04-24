import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageCoordinationBlock(nn.Module):
    def __init__(
            self,
            vit_token_count=50,
            vit_dim=1024,
            spatial_size=512,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1
            ):
        super().__init__()

        self.spatial_size = spatial_size
        self.rcan_proj = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.ffa_proj = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, spatial_size * spatial_size),
            nn.ReLU(inplace=True)
        )
        self.vit_conv = nn.Sequential(
            nn.Conv2d(vit_token_count, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
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
    
    def forward(self, x_rcan, x_ffa, x_vit):
        B = x_rcan.shape[0]

        # Upsample to match RCAN
        x_ffa = F.interpolate(x_ffa, size=x_rcan.shape[2:], mode='bilinear', align_corners=False)
        x_ffa = self.ffa_proj(x_ffa)

        x_vit = self.vit_proj(x_vit)
        x_vit = x_vit.view(B, -1, self.spatial_size, self.spatial_size)
        x_vit = self.vit_conv(x_vit) 
        
        x_rcan = self.rcan_proj(x_rcan)
        
        # Each model passes through separate layers
        x_vit = self.groups[0](x_vit)
        x_ffa = self.groups[1](x_ffa)
        x_rcan = self.groups[2](x_rcan)

        # Concatenate the outputs from all three models
        x_fused = torch.cat([x_vit, x_ffa, x_rcan], dim=1)
        output = self.final_layers(x_fused)
        output = self.attention(output)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels=32, reduction=16):
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
