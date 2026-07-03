"""HW5 -- Denoising Diffusion Probabilistic Model (Ho et al. 2020), question 11-13.

Implements the DDPM training objective the assignment derives:

  * forward process q(x_t | x_0) = N(sqrt(abar_t) x_0, (1-abar_t) I);
  * epsilon-parameterised model eps_theta(x_t, t) trained with
        L = || eps - eps_theta(x_t, t) ||^2 ;
  * ancestral sampling to denoise from pure noise back to an image.

A compact time-conditioned U-Net-lite serves as eps_theta, trained on MNIST.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


class DiffusionSchedule:
    """Precomputes the alpha / alpha-bar quantities used by forward & reverse."""

    def __init__(self, T=200):
        self.T = T
        self.beta = make_beta_schedule(T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise):
        """Sample x_t from x_0 in closed form (question 7)."""
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the (integer) timestep."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.temb = nn.Linear(temb_dim, cout)
        self.norm1 = nn.GroupNorm(8, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """A small time-conditioned U-Net acting as eps_theta(x_t, t)."""

    def __init__(self, channels=1, base=32, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.temb_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.in_conv = nn.Conv2d(channels, base, 3, padding=1)
        self.down1 = ConvBlock(base, base, temb_dim)
        self.down2 = ConvBlock(base, base * 2, temb_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid = ConvBlock(base * 2, base * 2, temb_dim)
        self.up2 = ConvBlock(base * 4, base, temb_dim)
        self.up1 = ConvBlock(base * 2, base, temb_dim)
        self.out_conv = nn.Conv2d(base, channels, 3, padding=1)

    def forward(self, x, t):
        temb = self.temb_mlp(timestep_embedding(t, self.temb_dim))
        h0 = self.in_conv(x)              # (base, 28, 28)
        h1 = self.down1(h0, temb)         # (base, 28, 28)
        h2 = self.down2(self.pool(h1), temb)  # (2base, 14, 14)
        m = self.mid(h2, temb)            # (2base, 14, 14)
        u2 = self.up2(torch.cat([m, h2], 1), temb)   # (base, 14, 14)
        u2 = F.interpolate(u2, scale_factor=2, mode="nearest")  # (base, 28, 28)
        u1 = self.up1(torch.cat([u2, h1], 1), temb)  # (base, 28, 28)
        return self.out_conv(u1)


def diffusion_loss(model, sched, x0):
    """L = || eps - eps_theta(x_t, t) ||^2, t sampled uniformly (question 11)."""
    B = x0.size(0)
    t = torch.randint(0, sched.T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = sched.q_sample(x0, t, noise)
    pred = model(x_t, t)
    return F.mse_loss(pred, noise)


@torch.no_grad()
def sample(model, sched, n=16, image_size=28, channels=1, capture_ts=None):
    """Ancestral sampling from noise; optionally capture intermediate x_t."""
    device = next(model.parameters()).device
    x = torch.randn(n, channels, image_size, image_size, device=device)
    captures = {}
    capture_ts = set(capture_ts or [])
    for t in reversed(range(sched.T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        alpha = sched.alpha[t]
        alpha_bar = sched.alpha_bar[t]
        beta = sched.beta[t]
        coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        mean = (x - coef * eps) / torch.sqrt(alpha)
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
        if t in capture_ts:
            captures[t] = x.clamp(0, 1).cpu()
    return x.clamp(0, 1), captures
