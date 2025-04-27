import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1))
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bnhd,bmhd->bhnm', q * scale, k)
        attn = attn.softmax(-1)
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v).reshape(B, C, H, W)
        return self.proj(out) + x_in

class ResidualBlock(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, time_emb_dim=32):
        super().__init__()
        self.time_layer = nn.Sequential(
            nn.Linear(time_emb_dim, out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.residual_conv = nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, time):
        out = self.conv1(x)
        time_emb = self.time_layer(time).view(time.shape[0], -1, 1, 1)
        out = out + time_emb
        out = self.relu(out)
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_channel, out_channel, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, time_emb):
        x = self.res1(x, time_emb)
        return x, self.pool(x)

class UpBlock(nn.Module):
    def __init__(self, x_channel, skip_channel, out_channel, time_emb_dim):
        super().__init__()
        self.conv_up = nn.ConvTranspose2d(x_channel, out_channel, kernel_size=2, stride=2)
        self.res1 = ResidualBlock(out_channel + skip_channel, out_channel, time_emb_dim)

    def forward(self, x, skip, time_emb):
        x = self.conv_up(x)
        x = torch.cat([x, skip], dim=1)
        out = self.res1(x, time_emb)
        return out

class DDPMNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        hidden_dim=32,
        time_emb_dim=32,
        kernel_size=3,
        padding=1
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_layer = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        self.hazy_proj = nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channel * 2, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.down1 = DownBlock(hidden_dim, hidden_dim * 2, time_emb_dim)
        self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4, time_emb_dim)

        self.mid_res = ResidualBlock(hidden_dim * 4, hidden_dim * 4, time_emb_dim)
        self.mid_attn = AttentionBlock(hidden_dim * 4)

        self.up2 = UpBlock(hidden_dim * 4, hidden_dim * 4, hidden_dim * 2, time_emb_dim)
        self.up1 = UpBlock(hidden_dim * 2, hidden_dim * 2, hidden_dim, time_emb_dim)

        self.conv2 = nn.Conv2d(hidden_dim, out_channel, kernel_size=1)

        self.cond_encoder = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

    def forward(self, x_t, x_hazy, time):
        time_embeddings = self.time_embed(time).to(x_t.device)
        time_embeddings = self.time_layer(time_embeddings)

        hazy_feat = self.cond_encoder(x_hazy)
        hazy_feat = self.hazy_proj(hazy_feat)

        x = self.conv1(torch.cat([x_t, x_hazy], dim=1))

        skip1, x = self.down1(x, time_embeddings)
        skip2, x = self.down2(x, time_embeddings)

        if hazy_feat.shape[-2:] != x.shape[-2:]:
            hazy_feat = F.interpolate(hazy_feat, size=x.shape[-2:], mode="bilinear", align_corners=False)

        x = x + hazy_feat

        x = self.mid_res(x, time_embeddings)
        x = self.mid_attn(x)

        x = self.up2(x, skip2, time_embeddings)
        x = self.up1(x, skip1, time_embeddings)

        out = self.conv2(x)
        return out

class Diffusion:
    def __init__(self, timesteps=10, beta=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, beta, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        device = x_0.device
        alpha_hats = self.alpha_hats.to(device)
  
        noise = torch.randn_like(x_0)
        sqrt_ah = torch.sqrt(alpha_hats[t])[:, None, None, None]
        sqrt_oma = torch.sqrt(1 - alpha_hats[t])[:, None, None, None]
        x_t = sqrt_ah * x_0 + sqrt_oma * noise
        
        return x_t, noise
    
    @torch.no_grad()
    def sample(self, model, x_hazy, device):
        model.eval()
        x = torch.randn_like(x_hazy, device=device)
        for t in reversed(range(self.timesteps)):
            time_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=device)
            beta = self.betas[t].to(device)
            alpha = self.alphas[t].to(device)
            alpha_hat = self.alpha_hats[t].to(device)
            sqrt_ah = torch.sqrt(alpha_hat)
            sqrt_oma = torch.sqrt(1.0 - alpha_hat)

            v_pred = model(x, x_hazy, time_tensor)
            x0_pred = sqrt_ah * x - sqrt_oma * v_pred
            eps_pred = (v_pred + sqrt_oma * x0_pred) / sqrt_ah
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_hat)
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = coef1 * (x - coef2 * eps_pred) + torch.sqrt(beta) * noise

        B = x.size(0)
        flat = x.abs().view(B, -1)
        s = torch.quantile(flat, 0.995, dim=1).view(B, 1, 1, 1)
        s = s.clamp_max(1.0)
        x = x.clamp(-s, s) / s
        return x.clamp(0.0, 1.0)
