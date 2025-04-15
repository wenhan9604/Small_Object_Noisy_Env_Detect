import torch
import torch.nn as nn


class FFANet(nn.Module):
    def __init__(self, num_groups=4, num_blocks=2, in_channels=3, out_channels=3, hidden_dim=64, kernel_size=3, padding=1, remove_global_skip_connection=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.groups = nn.ModuleList([
            nn.Sequential(*[
                ResidualBlockWithAttention(hidden_dim, hidden_dim) for _ in range(num_blocks)
                ])
            for _ in range(num_groups)
        ])
        self.remove_skip = remove_global_skip_connection
        self.attention = ResidualBlockWithAttention(in_channels=hidden_dim*num_groups, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        outputs = []
        for group in self.groups:
            x = group(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.attention(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Modification to remove the global skip connection as mentioned in the paper
        if not self.remove_skip:
            x += residual
        return x

class ResidualBlockWithAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.attention = FeatureAttention(in_channels=out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out += residual
        out = self.conv2(out)
        out = self.attention(out)
        out += residual
        return out

class FeatureAttention(nn.Module):
    def __init__(self, in_channels=64, reduction=16):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)

        # Channel attention
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Pixel attention
        self.pixel_fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_fc(x)
        x = x * channel_attention
        pixel_attention = self.pixel_fc(x)
        x = x * pixel_attention
        return x