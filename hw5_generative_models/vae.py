"""HW5 -- Variational Autoencoder (question 4-6).

Implements the VAE encoder (outputs mean and log-variance), the reparameterisation
trick, an MLP decoder, and the ELBO objective

    L = E_q[log p(x|z)]  -  KL(q(z|x) || N(0, I))

trained on FashionMNIST. Also provides ``plot_latents`` which sweeps a pair of
latent dimensions on a grid (others held at zero) and decodes, reproducing the
latent-traversal figure the assignment asks for.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, in_dim=784, hidden=400, latent=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim), nn.Sigmoid(),
        )
        self.latent = latent
        self.in_dim = in_dim

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def elbo_loss(recon, x, mu, logvar):
    """Negative ELBO (to minimise). Reconstruction = binary cross entropy;
    KL between q(z|x)=N(mu, diag(exp(logvar))) and N(0, I) has a closed form.
    """
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl) / x.size(0), recon_loss / x.size(0), kl / x.size(0)


@torch.no_grad()
def plot_latents(model, dim_i, dim_j, grid=10, span=2.5, image_size=28):
    """Decode a grid x grid sweep of latent dims (i, j); others set to zero."""
    import numpy as np
    vals = torch.linspace(-span, span, grid)
    canvas = np.zeros((grid * image_size, grid * image_size), dtype=np.float32)
    for a, vi in enumerate(vals):
        for b, vj in enumerate(vals):
            z = torch.zeros(1, model.latent)
            z[0, dim_i] = vi
            z[0, dim_j] = vj
            img = model.decode(z).view(image_size, image_size).numpy()
            canvas[a * image_size:(a + 1) * image_size,
                   b * image_size:(b + 1) * image_size] = img
    return canvas
