"""HW2 -- learning-rate transfer across width via spectral-norm scaling.

Reproduces the assignment's key empirical claim: with a *spectral-scaled* update
(semi-orthogonal init  W_k = sqrt(d_k/d_{k-1}) M_k, and the sign-gradient step
normalised by its spectral norm and scaled by sqrt(d_k/d_{k-1})), the optimal
learning rate found at a small width also works at a large width -- i.e. the LR
"transfers". A naive per-element step does *not* transfer.

We sweep the LR at a small width, pick the best, then train a much wider network
and show the same LR remains (near) optimal.  Real training on a synthetic but
non-trivial regression target so the whole thing runs in seconds on CPU.

Outputs: results/hw2/lr_transfer.png and lr_transfer.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from common.utils import set_seed


def make_task(n=2048, din=32, dout=1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, din)).astype(np.float32)
    # a fixed non-linear target
    W1 = rng.standard_normal((din, 64)).astype(np.float32) / np.sqrt(din)
    W2 = rng.standard_normal((64, dout)).astype(np.float32) / np.sqrt(64)
    Y = np.tanh(X @ W1) @ W2
    return torch.tensor(X), torch.tensor(Y)


class MLP(nn.Module):
    def __init__(self, din, width, depth, dout=1):
        super().__init__()
        dims = [din] + [width] * depth + [dout]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)]
        )
        self.dims = dims
        self._spectral_init()

    def _spectral_init(self):
        # W_k = sqrt(d_k / d_{k-1}) * (semi-orthogonal M_k)
        for lin in self.linears:
            dout_, din_ = lin.weight.shape
            M = torch.empty(dout_, din_)
            nn.init.orthogonal_(M)
            with torch.no_grad():
                lin.weight.copy_(np.sqrt(dout_ / din_) * M)

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = torch.relu(x)
        return x


def spectral_norm_2d(W):
    return torch.linalg.matrix_norm(W, ord=2)


def train_once(width, depth, lr, steps, spectral_update, seed=0):
    set_seed(seed)
    X, Y = make_task(seed=seed)
    model = MLP(X.shape[1], width, depth)
    lossf = nn.MSELoss()
    for _ in range(steps):
        pred = model(X)
        loss = lossf(pred, Y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for lin in model.linears:
                g = lin.weight.grad
                dout_, din_ = lin.weight.shape
                if spectral_update:
                    # sign gradient normalised by its spectral norm, scaled by
                    # sqrt(d_k / d_{k-1}) -- the width-robust update rule.
                    s = torch.sign(g)
                    step = np.sqrt(dout_ / din_) * s / (spectral_norm_2d(s) + 1e-9)
                else:
                    # naive elementwise SGD (does not transfer across width)
                    step = g
                lin.weight -= lr * step
    with torch.no_grad():
        final = lossf(model(X), Y).item()
    return final


def sweep(width, depth, lrs, steps, spectral_update):
    return {lr: train_once(width, depth, lr, steps, spectral_update) for lr in lrs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--small-width", type=int, default=64)
    ap.add_argument("--large-width", type=int, default=1024)
    ap.add_argument("--outdir", default="results/hw2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    lrs = [10 ** e for e in np.linspace(-3, 0.5, 12)]

    def nan_to_inf(d):
        return {k: (v if np.isfinite(v) else float("inf")) for k, v in d.items()}

    results = {}
    for name, spectral in [("spectral", True), ("naive", False)]:
        small = nan_to_inf(sweep(args.small_width, args.depth, lrs, args.steps, spectral))
        large = nan_to_inf(sweep(args.large_width, args.depth, lrs, args.steps, spectral))
        best_small_lr = min(small, key=small.get)
        best_large_lr = min(large, key=large.get)
        # Transfer quality: take the LR that is best at SMALL width and ask how
        # much worse it is (relative to that width's own best) when applied at
        # LARGE width. A well-transferring rule keeps this ratio near 1 and never
        # diverges; a non-transferring rule blows up (inf).
        large_at_small_best = large[best_small_lr]
        large_best = large[best_large_lr]
        transfer_ratio = (large_at_small_best / large_best
                          if np.isfinite(large_at_small_best) else float("inf"))
        results[name] = {
            "small": small, "large": large,
            "best_small_lr": best_small_lr, "best_large_lr": best_large_lr,
            "transfer_gap_log10": abs(np.log10(best_small_lr) - np.log10(best_large_lr)),
            "large_loss_at_small_best_lr": large_at_small_best,
            "large_best_loss": large_best,
            "transfer_ratio": transfer_ratio,
        }
        print(f"[{name}] best-small LR={best_small_lr:.3g} -> large-width loss="
              f"{large_at_small_best:.4g} (large-best={large_best:.4g}, "
              f"ratio={transfer_ratio:.2f})")

    # Figure: loss vs LR for both widths, spectral vs naive.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    cap = 10.0  # plot ceiling so divergences (inf) are visible, not dropped

    def clip(vals):
        return [min(v, cap) for v in vals]

    for j, name in enumerate(["spectral", "naive"]):
        s = results[name]["small"]; l = results[name]["large"]
        ax[j].plot(list(s.keys()), clip(list(s.values())), "o-",
                   label=f"width {args.small_width}")
        ax[j].plot(list(l.keys()), clip(list(l.values())), "s-",
                   label=f"width {args.large_width}")
        ax[j].axhline(cap, ls=":", c="red", alpha=0.5, label="diverged (capped)")
        ax[j].set_xscale("log"); ax[j].set_yscale("log")
        ax[j].set_title(f"{name} update"); ax[j].set_xlabel("learning rate")
        ax[j].legend(fontsize=8); ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("final MSE")
    fig.suptitle("HW2: LR transfer across width (spectral scaling transfers, naive does not)")
    fig.tight_layout()
    figpath = os.path.join(args.outdir, "lr_transfer.png")
    fig.savefig(figpath, dpi=110)

    # JSON-friendly (inf -> string so it round-trips)
    def jval(v):
        return v if np.isfinite(v) else "inf"

    def keyfmt(d):
        return {f"{k:.4g}": jval(v) for k, v in d.items()}

    dump = {name: {"small": keyfmt(r["small"]), "large": keyfmt(r["large"]),
                   "best_small_lr": r["best_small_lr"], "best_large_lr": r["best_large_lr"],
                   "transfer_gap_log10": r["transfer_gap_log10"],
                   "large_loss_at_small_best_lr": jval(r["large_loss_at_small_best_lr"]),
                   "large_best_loss": jval(r["large_best_loss"]),
                   "transfer_ratio": jval(r["transfer_ratio"])}
            for name, r in results.items()}
    with open(os.path.join(args.outdir, "lr_transfer.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Saved {figpath}")


if __name__ == "__main__":
    main()
