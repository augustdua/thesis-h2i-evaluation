"""ADM-style U-Net for the variance-exploding H&E-to-IHC bridge.

Six-channel input (bridge state concatenated with H&E) and a three-channel clean-IHC prediction.
Copied from bbdm_BCI/ve_bridge/ve_bridge.py. The bridge sampler, stain metrics, and training loop are not included.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, ch)


def zero(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_layers = nn.Sequential(norm(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * cout))
        self.out_norm = norm(cout)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), zero(nn.Conv2d(cout, cout, 3, padding=1)))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class Attention(nn.Module):
    def __init__(self, ch: int, head_ch: int = 64):
        super().__init__()
        self.heads = max(1, ch // head_ch)
        self.norm = norm(ch)
        self.qkv = nn.Conv1d(ch, 3 * ch, 1)
        self.proj = zero(nn.Conv1d(ch, ch, 1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        b, c, hh, ww = x.shape
        qkv = self.qkv(self.norm(x).reshape(b, c, -1))
        q, k, v = qkv.reshape(b, 3, self.heads, c // self.heads, -1).permute(1, 0, 2, 4, 3)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(b, c, -1)
        return x + self.proj(h).reshape(b, c, hh, ww)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, emb=None):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, emb=None):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Seq(nn.ModuleList):
    def forward(self, x, emb):
        for m in self:
            x = m(x, emb)
        return x


@dataclass
class UNetConfig:
    in_ch: int = 6
    out_ch: int = 3
    base: int = 64
    mults: tuple[int, ...] = (1, 2, 3, 4, 4)
    num_res: int = 2
    attn_res: tuple[int, ...] = (16,)
    resolution: int = 256
    dropout: float = 0.0


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig = UNetConfig()):
        super().__init__()
        self.cfg = cfg
        emb_dim = 4 * cfg.base
        self.time = nn.Sequential(nn.Linear(cfg.base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.inp = nn.Conv2d(cfg.in_ch, cfg.base, 3, padding=1)
        self.down = nn.ModuleList()
        skips = [cfg.base]
        ch, res = cfg.base, cfg.resolution
        for lvl, m in enumerate(cfg.mults):
            for _ in range(cfg.num_res):
                layers = [ResBlock(ch, cfg.base * m, emb_dim, cfg.dropout)]
                ch = cfg.base * m
                if res in cfg.attn_res:
                    layers.append(Attention(ch))
                self.down.append(Seq(layers))
                skips.append(ch)
            if lvl != len(cfg.mults) - 1:
                self.down.append(Seq([Down(ch)]))
                skips.append(ch)
                res //= 2
        self.mid = Seq([ResBlock(ch, ch, emb_dim, cfg.dropout), Attention(ch), ResBlock(ch, ch, emb_dim, cfg.dropout)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(cfg.mults))):
            for i in range(cfg.num_res + 1):
                layers = [ResBlock(ch + skips.pop(), cfg.base * m, emb_dim, cfg.dropout)]
                ch = cfg.base * m
                if res in cfg.attn_res:
                    layers.append(Attention(ch))
                if lvl != 0 and i == cfg.num_res:
                    layers.append(Up(ch))
                    res *= 2
                self.up.append(Seq(layers))
        self.out = nn.Sequential(norm(ch), nn.SiLU(), zero(nn.Conv2d(ch, cfg.out_ch, 3, padding=1)))

    def forward(self, x: torch.Tensor, t1000: torch.Tensor) -> torch.Tensor:
        emb = self.time(timestep_embedding(t1000, self.cfg.base))
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            h = m(h, emb)
            hs.append(h)
        h = self.mid(h, emb)
        for m in self.up:
            h = m(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)
