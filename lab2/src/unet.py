"""Configurable U-Net for DDPM with sinusoidal time embedding and optional self-attention.

Inspired by the standard DDPM reference (Ho et al., 2020) and MiniDiffusion repo.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over spatial positions."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, c, h * w).unbind(dim=1)  # (b, c, hw)
        attn = torch.softmax(torch.einsum("bci,bcj->bij", q, k) * (c ** -0.5), dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# ----------------------------------------------------------------------
# U-Net
# ----------------------------------------------------------------------
class UNet(nn.Module):
    """Configurable U-Net.

    Parameters
    ----------
    img_channels  : 1 for MNIST, 3 for CIFAR-10/Flowers
    base_channels : starting number of channels (32 for small, 64 for default)
    channel_mults : multipliers per level, e.g. (1, 2, 4)
    num_res       : ResBlocks per level
    attn_at       : list of spatial resolutions at which to insert self-attention,
                    e.g. [8] inserts attention in the 8x8 bottleneck.
    """

    def __init__(
        self,
        img_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple = (1, 2, 4),
        num_res: int = 1,
        attn_at: tuple | list = (),
        time_dim_mult: int = 4,
        dropout: float = 0.1,
        img_size: int = 32,
    ):
        super().__init__()
        time_dim = base_channels * time_dim_mult
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chs = [base_channels]
        ch = base_channels
        res = img_size
        for i, m in enumerate(channel_mults):
            out_ch = base_channels * m
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for j in range(num_res):
                level_blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                if res in attn_at:
                    level_attns.append(SelfAttention(ch))
                else:
                    level_attns.append(nn.Identity())
                chs.append(ch)
            self.down_blocks.append(level_blocks)
            self.down_attns.append(level_attns)
            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                chs.append(ch)
                res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch) if res in attn_at else nn.Identity()
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, m in enumerate(reversed(channel_mults)):
            out_ch = base_channels * m
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for j in range(num_res + 1):
                level_blocks.append(ResBlock(ch + chs.pop(), out_ch, time_dim, dropout))
                ch = out_ch
                if res in attn_at:
                    level_attns.append(SelfAttention(ch))
                else:
                    level_attns.append(nn.Identity())
            self.up_blocks.append(level_blocks)
            self.up_attns.append(level_attns)
            if i != len(channel_mults) - 1:
                self.upsamples.append(Upsample(ch))
                res *= 2
            else:
                self.upsamples.append(nn.Identity())

        # Out
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.input_conv(x)
        hs = [h]

        # Encoder
        for i, (blocks, attns) in enumerate(zip(self.down_blocks, self.down_attns)):
            for block, attn in zip(blocks, attns):
                h = block(h, t_emb)
                h = attn(h)
                hs.append(h)
            h = self.downsamples[i](h)
            if not isinstance(self.downsamples[i], nn.Identity):
                hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for i, (blocks, attns) in enumerate(zip(self.up_blocks, self.up_attns)):
            for block, attn in zip(blocks, attns):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, t_emb)
                h = attn(h)
            h = self.upsamples[i](h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
