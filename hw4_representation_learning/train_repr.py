"""HW4 -- train reconstruction- vs similarity-based encoders and probe them.

Trains:
  * an autoencoder (reconstruction objective);
  * a contrastive encoder under three augmentation regimes:
      - recolor   -> should become invariant to color, sensitive to shape/location
      - translate -> should become invariant to location
      - jitter    -> preserves all factors

Then probes each learned representation with k-NN factor prediction (shape /
color) computed on a held-out validation split. This makes the assignment's
qualitative "nearest neighbour" observations quantitative: e.g. a recolor-
augmented encoder should have near-chance color accuracy but high shape accuracy.

Outputs: results/hw4/repr_probe.json, results/hw4/nn_grid.png
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
from hw4_representation_learning.shapes_dataset import (
    make_dataset, aug_recolor, aug_translate, aug_jitter,
)
from hw4_representation_learning.models import (
    Encoder, Decoder, ae_loss, contrastive_loss,
)

AUGS = {"recolor": aug_recolor, "translate": aug_translate, "jitter": aug_jitter}


def knn_accuracy(train_z, train_lab, val_z, val_lab, k=5):
    """k-NN classification accuracy of a factor from representation distances."""
    # cosine distance
    tz = train_z / (np.linalg.norm(train_z, axis=1, keepdims=True) + 1e-9)
    vz = val_z / (np.linalg.norm(val_z, axis=1, keepdims=True) + 1e-9)
    sims = vz @ tz.T                          # (Nval, Ntrain)
    nn_idx = np.argsort(-sims, axis=1)[:, :k]
    correct = 0
    for i in range(vz.shape[0]):
        votes = train_lab[nn_idx[i]]
        pred = np.bincount(votes).argmax()
        correct += int(pred == val_lab[i])
    return correct / vz.shape[0]


def embed(encoder, imgs, batch=256):
    encoder.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            xb = torch.tensor(imgs[i:i + batch])
            outs.append(encoder(xb).numpy())
    return np.concatenate(outs, 0)


def train_autoencoder(imgs, d, epochs, lr):
    enc = Encoder(d=d, normalize=False)
    dec = Decoder(d=d)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    loader = DataLoader(TensorDataset(torch.tensor(imgs)),
                        batch_size=128, shuffle=True)
    for ep in range(epochs):
        enc.train(); dec.train()
        tot = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            loss = ae_loss(dec(enc(xb)), xb)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"  [AE] epoch {ep + 1}/{epochs} loss={tot / len(loader):.4f}")
    return enc


def train_contrastive(imgs, aug_fn, d, epochs, lr, tau=0.07, seed=0):
    rng = np.random.default_rng(seed)
    enc = Encoder(d=d, normalize=True)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    n = len(imgs)
    idx = np.arange(n)
    bs = 128
    for ep in range(epochs):
        enc.train()
        rng.shuffle(idx)
        tot = 0.0
        nb = 0
        for start in range(0, n - bs, bs):
            sel = idx[start:start + bs]
            v1 = np.stack([aug_fn(imgs[i], rng) for i in sel])
            v2 = np.stack([aug_fn(imgs[i], rng) for i in sel])
            z1 = enc(torch.tensor(v1))
            z2 = enc(torch.tensor(v2))
            loss = contrastive_loss(z1, z2, tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [contrastive/{aug_fn.__name__}] epoch {ep + 1}/{epochs} "
              f"loss={tot / nb:.4f}")
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--ae-epochs", type=int, default=12)
    ap.add_argument("--contrastive-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", default="results/hw4")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(0)

    print("Building shapes dataset ...")
    train_imgs, train_fac = make_dataset(args.n_train, seed=0)
    val_imgs, val_fac = make_dataset(args.n_val, seed=1)

    results = {}
    t0 = time.time()

    def probe(enc, name):
        tz = embed(enc, train_imgs)
        vz = embed(enc, val_imgs)
        shape_acc = knn_accuracy(tz, train_fac["shape"], vz, val_fac["shape"])
        color_acc = knn_accuracy(tz, train_fac["color"], vz, val_fac["color"])
        results[name] = {"knn_shape_acc": round(shape_acc, 4),
                         "knn_color_acc": round(color_acc, 4)}
        print(f"  {name}: shape_acc={shape_acc:.3f}  color_acc={color_acc:.3f}")
        return enc

    print("Training autoencoder ...")
    ae_enc = train_autoencoder(train_imgs, args.d, args.ae_epochs, args.lr)
    probe(ae_enc, "autoencoder")

    for aug_name, aug_fn in AUGS.items():
        print(f"Training contrastive encoder (aug={aug_name}) ...")
        enc = train_contrastive(train_imgs, aug_fn, args.d,
                                args.contrastive_epochs, args.lr)
        probe(enc, f"contrastive_{aug_name}")

    elapsed = time.time() - t0
    results["_meta"] = {
        "n_train": args.n_train, "n_val": args.n_val, "d": args.d,
        "chance_shape": round(1 / 3, 4), "chance_color": 0.25,
        "seconds": round(elapsed, 1),
    }

    # Nearest-neighbour visual grid for the recolor encoder + AE (val set).
    _save_nn_grid(ae_enc, val_imgs, os.path.join(args.outdir, "nn_grid.png"))

    with open(os.path.join(args.outdir, "repr_probe.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


def _save_nn_grid(encoder, imgs, path, n_query=6, k=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = embed(encoder, imgs)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    sims = z @ z.T
    np.fill_diagonal(sims, -1)
    fig, ax = plt.subplots(n_query, k + 1, figsize=(2 * (k + 1), 2 * n_query))
    disp = imgs.transpose(0, 2, 3, 1).clip(0, 1)
    for r in range(n_query):
        q = r * (len(imgs) // n_query)
        ax[r, 0].imshow(disp[q]); ax[r, 0].set_title("query", fontsize=8)
        ax[r, 0].axis("off")
        for j, nn in enumerate(np.argsort(-sims[q])[:k]):
            ax[r, j + 1].imshow(disp[nn]); ax[r, j + 1].axis("off")
    fig.suptitle("HW4 autoencoder: nearest neighbours in representation space")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


if __name__ == "__main__":
    main()
