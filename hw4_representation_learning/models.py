"""HW4 -- encoder/decoder and the two representation-learning objectives.

  * ``Encoder`` -- a small conv net mapping a 64x64 image to a d-dim latent;
  * ``Decoder`` -- transpose-conv net back to an image (for the autoencoder);
  * ``ae_loss`` -- mean-squared reconstruction loss (question 7b);
  * ``contrastive_loss`` -- InfoNCE with in-batch negatives and l2-normalised
    features and temperature tau (question 8a).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, d=16, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),   # 32x32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 16x16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(), # 8x8
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),# 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, d),
        )

    def forward(self, x):
        z = self.net(x)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z


class Decoder(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.fc = nn.Linear(d, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),    # 32
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),  # 64
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        return self.net(x)


def ae_loss(recon, target):
    """Mean squared reconstruction error (question 7b)."""
    return F.mse_loss(recon, target)


def contrastive_loss(z, z_pos, tau=0.07):
    """InfoNCE with in-batch negatives.

    z, z_pos: (B, d) l2-normalised encodings of the two augmented views.
    For anchor i, the positive is z_pos[i] and negatives are z_pos[j!=i].
    Symmetrised over both directions.
    """
    logits = z @ z_pos.t() / tau                 # (B, B)
    labels = torch.arange(z.size(0), device=z.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))
