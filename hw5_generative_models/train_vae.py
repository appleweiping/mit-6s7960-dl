"""HW5 -- train the VAE on FashionMNIST and render latent traversals.

Outputs: results/hw5/vae_metrics.json, results/hw5/vae_recon.png,
         results/hw5/vae_latents.png
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.utils import set_seed
from hw5_generative_models.data import load_fashion_mnist
from hw5_generative_models.vae import VAE, elbo_loss, plot_latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--latent", type=int, default=20)
    ap.add_argument("--outdir", default="results/hw5")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(0)

    xtr, xte = load_fashion_mnist()
    xtr_flat = xtr.reshape(len(xtr), -1)
    loader = DataLoader(TensorDataset(torch.tensor(xtr_flat)),
                        batch_size=args.batch_size, shuffle=True)

    model = VAE(latent=args.latent)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    history = []
    for ep in range(args.epochs):
        model.train()
        tot = rec = kl = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, r, k = elbo_loss(recon, xb, mu, logvar)
            loss.backward(); opt.step()
            tot += loss.item(); rec += r.item(); kl += k.item()
        n = len(loader)
        history.append({"elbo_neg": tot / n, "recon": rec / n, "kl": kl / n})
        print(f"epoch {ep + 1}/{args.epochs}  -ELBO={tot / n:.2f} "
              f"recon={rec / n:.2f} KL={kl / n:.2f}")
    elapsed = time.time() - t0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reconstructions on test set
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(xte[:8].reshape(8, -1))
        recon, _, _ = model(xb)
    fig, ax = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        ax[0, i].imshow(xte[i], cmap="gray"); ax[0, i].axis("off")
        ax[1, i].imshow(recon[i].view(28, 28), cmap="gray"); ax[1, i].axis("off")
    ax[0, 0].set_title("input", fontsize=8); ax[1, 0].set_title("recon", fontsize=8)
    fig.suptitle("HW5 VAE reconstructions (FashionMNIST)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "vae_recon.png"), dpi=110)

    # latent traversal grid for dims (0, 1)
    canvas = plot_latents(model, 0, 1, grid=10, span=2.5)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(canvas, cmap="gray"); ax2.axis("off")
    ax2.set_title("HW5 VAE latent traversal (dims 0 vs 1)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "vae_latents.png"), dpi=110)

    metrics = {
        "epochs": args.epochs, "latent": args.latent,
        "final_neg_elbo": history[-1]["elbo_neg"],
        "final_recon": history[-1]["recon"], "final_kl": history[-1]["kl"],
        "history": history, "seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.outdir, "vae_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in
                      ["final_neg_elbo", "final_recon", "final_kl", "seconds"]}, indent=2))


if __name__ == "__main__":
    main()
