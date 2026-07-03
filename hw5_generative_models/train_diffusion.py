"""HW5 -- train the DDPM on MNIST and render the reverse-diffusion figure.

Reproduces the assignment's diffusion task: train eps_theta with the simple
noise-prediction loss, then produce samples and a 6-panel figure of the reverse
process x_t for t in {200, 100, 50, 20, 10, 0}.

Outputs: results/hw5/diffusion_metrics.json, results/hw5/diffusion_samples.png,
         results/hw5/diffusion_reverse.png
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
from hw5_generative_models.data import load_mnist
from hw5_generative_models.diffusion import (
    DiffusionSchedule, UNet, diffusion_loss, sample,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--subset", type=int, default=12000)
    ap.add_argument("--outdir", default="results/hw5")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(0)

    xtr, _ = load_mnist()
    if args.subset > 0:
        xtr = xtr[:args.subset]
    loader = DataLoader(TensorDataset(torch.tensor(xtr)),
                        batch_size=args.batch_size, shuffle=True)

    sched = DiffusionSchedule(T=args.timesteps)
    model = UNet(channels=1, base=32)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    losses = []
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            loss = diffusion_loss(model, sched, xb)
            loss.backward(); opt.step()
            tot += loss.item()
        losses.append(tot / len(loader))
        print(f"epoch {ep + 1}/{args.epochs}  loss={tot / len(loader):.4f}")
    elapsed = time.time() - t0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 16 final samples
    capture_ts = [t for t in [200, 100, 50, 20, 10, 0] if t < args.timesteps]
    if args.timesteps - 1 not in capture_ts:
        capture_ts = [args.timesteps - 1] + capture_ts
    imgs, captures = sample(model, sched, n=16, capture_ts=capture_ts)
    grid = imgs.numpy().reshape(4, 4, 28, 28)
    fig, ax = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(4):
        for j in range(4):
            ax[i, j].imshow(grid[i, j], cmap="gray"); ax[i, j].axis("off")
    fig.suptitle("HW5 DDPM samples (MNIST)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "diffusion_samples.png"), dpi=110)

    # reverse-process panel: one image at descending t
    steps = sorted([t for t in [200, 100, 50, 20, 10, 0] if t in captures], reverse=True)
    fig2, ax2 = plt.subplots(1, len(steps), figsize=(2 * len(steps), 2.4))
    for k, t in enumerate(steps):
        ax2[k].imshow(captures[t][0, 0].numpy(), cmap="gray")
        ax2[k].set_title(f"t={t}"); ax2[k].axis("off")
    fig2.suptitle("HW5 DDPM reverse diffusion x_t")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "diffusion_reverse.png"), dpi=110)

    metrics = {
        "epochs": args.epochs, "timesteps": args.timesteps,
        "subset": args.subset, "final_loss": losses[-1],
        "loss_curve": losses, "seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.outdir, "diffusion_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"final_loss": losses[-1], "seconds": metrics["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
