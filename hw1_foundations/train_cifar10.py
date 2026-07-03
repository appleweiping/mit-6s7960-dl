"""HW1 -- train the from-scratch MLP on CIFAR-10 and log learning curves.

This exercises the hand-written Linear/ReLU/CrossEntropy forward+backward passes
(no autograd) plus a minimal SGD loop, and reproduces the two training-curve
questions of the assignment:

  * curve 1: plain training for a number of epochs (question 12);
  * curve 2: same model with random-crop data augmentation (questions 14-15).

Outputs (into ``results/hw1/``):
  * ``cifar10_curves.png``  -- train/val accuracy vs epoch, both settings;
  * ``cifar10_metrics.json`` -- final measured numbers.

Run:
    python -m hw1_foundations.train_cifar10 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common.utils import set_seed
from hw1_foundations.data import load_cifar10_flat
from hw1_foundations.modules import MLP


def iterate_minibatches(x, y, batch_size, rng, shuffle=True):
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        yield x[sel], y[sel]


def accuracy(model: MLP, x, y, batch_size=1000) -> float:
    correct = 0
    for xb, yb in iterate_minibatches(x, y, batch_size, None, shuffle=False):
        logits = model.forward(xb)
        correct += int((logits.argmax(axis=1) == yb).sum())
    return correct / x.shape[0]


def random_crop_flat(x_flat, rng, pad=4):
    """Random-crop augmentation on flattened CIFAR images (question 14)."""
    n = x_flat.shape[0]
    imgs = x_flat.reshape(n, 32, 32, 3)
    out = np.empty_like(imgs)
    padded = np.pad(imgs, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    for i in range(n):
        top = rng.integers(0, 2 * pad + 1)
        left = rng.integers(0, 2 * pad + 1)
        out[i] = padded[i, top:top + 32, left:left + 32, :]
    return out.reshape(n, -1)


def train(x_tr, y_tr, x_te, y_te, epochs, lr, batch_size, augment, seed=0):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP([3072, 512, 256, 10], seed=seed)
    hist = {"train_acc": [], "val_acc": [], "loss": []}
    for ep in range(epochs):
        x_ep = random_crop_flat(x_tr, rng) if augment else x_tr
        ep_loss = 0.0
        nb = 0
        for xb, yb in iterate_minibatches(x_ep, y_tr, batch_size, rng):
            logits = model.forward(xb)
            ep_loss += model.loss(logits, yb)
            model.backward()
            model.step(lr)
            nb += 1
        tr_acc = accuracy(model, x_tr, y_tr)
        va_acc = accuracy(model, x_te, y_te)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)
        hist["loss"].append(ep_loss / nb)
        print(f"  epoch {ep + 1:2d}/{epochs}  loss={ep_loss / nb:.3f}  "
              f"train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--outdir", default="results/hw1")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("Loading CIFAR-10 ...")
    x_tr, y_tr, x_te, y_te = load_cifar10_flat()
    print(f"  train {x_tr.shape}, test {x_te.shape}")

    t0 = time.time()
    print("Training curve 1 (no augmentation) ...")
    _, hist_plain = train(x_tr, y_tr, x_te, y_te, args.epochs, args.lr,
                          args.batch_size, augment=False)
    print("Training curve 2 (random-crop augmentation) ...")
    _, hist_aug = train(x_tr, y_tr, x_te, y_te, args.epochs, args.lr,
                        args.batch_size, augment=True)
    elapsed = time.time() - t0

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, args.epochs + 1)
    ax[0].plot(ep, hist_plain["train_acc"], label="train")
    ax[0].plot(ep, hist_plain["val_acc"], label="val")
    ax[0].set_title("Curve 1: no augmentation")
    ax[1].plot(ep, hist_aug["train_acc"], label="train")
    ax[1].plot(ep, hist_aug["val_acc"], label="val")
    ax[1].set_title("Curve 2: random-crop augmentation")
    for a in ax:
        a.set_xlabel("epoch"); a.set_ylabel("accuracy"); a.legend(); a.grid(alpha=0.3)
    fig.suptitle("HW1 scratch-MLP on CIFAR-10 (CPU)")
    fig.tight_layout()
    figpath = os.path.join(args.outdir, "cifar10_curves.png")
    fig.savefig(figpath, dpi=110)

    metrics = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "final_train_acc_plain": hist_plain["train_acc"][-1],
        "final_val_acc_plain": hist_plain["val_acc"][-1],
        "final_train_acc_aug": hist_aug["train_acc"][-1],
        "final_val_acc_aug": hist_aug["val_acc"][-1],
        "seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.outdir, "cifar10_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved figure to {figpath}")


if __name__ == "__main__":
    main()
