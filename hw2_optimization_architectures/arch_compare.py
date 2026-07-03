"""HW2 -- MLP vs CNN inductive bias on CIFAR-100.

Trains three architectures for the same number of epochs and compares validation
accuracy, reproducing the assignment's comparison:
  * MLP (width 128, depth 3)
  * MLP (width 128, depth 7)
  * CNN (4 conv layers + 3 MLP layers)

The point is inductive bias: the CNN's weight sharing / locality lets it beat
both MLPs, and simply making the MLP deeper does not close the gap.

Outputs: results/hw2/arch_compare.png and arch_compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.utils import set_seed
from hw2_optimization_architectures.cifar100_data import load_cifar100


class MLP(nn.Module):
    def __init__(self, width=128, depth=3, num_classes=100):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(3 * 32 * 32, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """4 conv layers followed by 3 MLP layers."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum())
            total += yb.numel()
    return correct / total


def train_model(model, train_loader, val_loader, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    accs = []
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        acc = evaluate(model, val_loader)
        accs.append(acc)
        print(f"    epoch {ep + 1}/{epochs}  val_acc={acc:.4f}")
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--subset", type=int, default=0,
                    help="if >0, use only this many train images (for speed)")
    ap.add_argument("--outdir", default="results/hw2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(0)

    xtr, ytr, xte, yte = load_cifar100()
    if args.subset > 0:
        xtr, ytr = xtr[:args.subset], ytr[:args.subset]
    train_ds = TensorDataset(torch.tensor(xtr), torch.tensor(ytr))
    val_ds = TensorDataset(torch.tensor(xte), torch.tensor(yte))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    configs = {
        "MLP_w128_d3": MLP(128, 3),
        "MLP_w128_d7": MLP(128, 7),
        "CNN_4conv_3mlp": CNN(),
    }

    results = {}
    t0 = time.time()
    for name, model in configs.items():
        print(f"Training {name} ...")
        accs = train_model(model, train_loader, val_loader, args.epochs, args.lr)
        results[name] = accs
        print(f"  {name} final val_acc={accs[-1]:.4f}")
    elapsed = time.time() - t0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, accs in results.items():
        ax.plot(range(1, len(accs) + 1), accs, "o-", label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy")
    ax.set_title("HW2: MLP vs CNN on CIFAR-100")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    figpath = os.path.join(args.outdir, "arch_compare.png")
    fig.savefig(figpath, dpi=110)

    summary = {
        "epochs": args.epochs, "lr": args.lr, "subset": args.subset,
        "seconds": round(elapsed, 1),
        "final_val_acc": {k: v[-1] for k, v in results.items()},
        "curves": results,
    }
    with open(os.path.join(args.outdir, "arch_compare.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["final_val_acc"], indent=2))
    print(f"Saved {figpath}")


if __name__ == "__main__":
    main()
