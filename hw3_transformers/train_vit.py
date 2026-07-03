"""HW3 -- train the from-scratch ViT on CIFAR-10 and save an attention heatmap.

Reproduces the assignment's ViT task: train to >50% validation accuracy and
visualise the class-token attention heatmap over patches for a few images.

Outputs: results/hw3/vit_metrics.json, results/hw3/vit_attention.png
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.utils import set_seed
from hw3_transformers.data import load_cifar10_images
from hw3_transformers.vit import VisionTransformer


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum())
            total += yb.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--outdir", default="results/hw3")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(0)

    xtr, ytr, xte, yte = load_cifar10_images()
    if args.subset > 0:
        xtr, ytr = xtr[:args.subset], ytr[:args.subset]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(xtr), torch.tensor(ytr)),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(xte), torch.tensor(yte)), batch_size=512)

    model = VisionTransformer(embed_dim=128, n_heads=4, n_layers=4, patch_size=4)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    accs = []
    for ep in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        acc = evaluate(model, val_loader)
        accs.append(acc)
        print(f"epoch {ep + 1}/{args.epochs}  val_acc={acc:.4f}")
    elapsed = time.time() - t0

    # Attention heatmaps for 10 validation images.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.arange(10)
    imgs = torch.tensor(xte[idx])
    heat = model.class_token_attention(imgs).numpy()
    # de-normalise for display
    mean = np.array([0.4914, 0.4822, 0.4465]); std = np.array([0.2470, 0.2435, 0.2616])
    disp = (xte[idx].transpose(0, 2, 3, 1) * std + mean).clip(0, 1)
    fig, ax = plt.subplots(2, 10, figsize=(18, 4))
    for i in range(10):
        ax[0, i].imshow(disp[i]); ax[0, i].axis("off")
        ax[1, i].imshow(heat[i], cmap="viridis"); ax[1, i].axis("off")
    ax[0, 0].set_ylabel("image"); ax[1, 0].set_ylabel("attn")
    fig.suptitle("HW3 ViT: class-token attention (avg over heads & layers)")
    fig.tight_layout()
    figpath = os.path.join(args.outdir, "vit_attention.png")
    fig.savefig(figpath, dpi=110)

    metrics = {
        "epochs": args.epochs, "lr": args.lr, "subset": args.subset,
        "final_val_acc": accs[-1], "val_acc_curve": accs,
        "seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.outdir, "vit_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"final_val_acc": accs[-1], "seconds": metrics["seconds"]}, indent=2))
    print(f"Saved {figpath}")


if __name__ == "__main__":
    main()
