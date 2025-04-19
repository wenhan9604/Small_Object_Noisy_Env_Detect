import torch
import torch.nn as nn


class FFANet(nn.Module):
    def __init__(self, num_groups=4, num_blocks=2, in_channels=3, out_channels=3, hidden_dim=32, kernel_size=3, remove_global_skip_connection=True):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, padding=self.padding)
        self.groups = nn.ModuleList([
            nn.Sequential(*[
                ResidualBlockWithAttention(hidden_dim) for _ in range(num_blocks)
                ])
            for _ in range(num_groups)
        ])
        self.remove_skip = remove_global_skip_connection
        self.fusion_conv = nn.Conv2d(hidden_dim * num_groups, hidden_dim, kernel_size=1)
        self.fusion_attention = FeatureAttention(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=kernel_size, padding=self.padding)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        outputs = []
        for group in self.groups:
            x = group(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.fusion_conv(x)
        x = self.fusion_attention(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Modification to remove the global skip connection as mentioned in the paper
        if not self.remove_skip:
            x += residual
        return x


class ResidualBlockWithAttention(nn.Module):
    def __init__(self, channels=32, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.attention = FeatureAttention(channels=channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.attention(out)
        out += residual
        return out

class FeatureAttention(nn.Module):
    def __init__(self, channels=32, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)

        # Channel attention
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Pixel attention
        self.pixel_fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_fc(x)
        x = x * channel_attention
        pixel_attention = self.pixel_fc(x)
        x = x * pixel_attention
        return x