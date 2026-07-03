"""HW5 datasets: FashionMNIST (VAE) and MNIST (diffusion), downloaded at runtime."""
from __future__ import annotations

import numpy as np
import torchvision

from common.utils import data_dir


def _load(cls, root):
    tr = cls(root=root, train=True, download=True)
    te = cls(root=root, train=False, download=True)
    xtr = (np.asarray(tr.data, dtype=np.float32) / 255.0)
    xte = (np.asarray(te.data, dtype=np.float32) / 255.0)
    return xtr, xte


def load_fashion_mnist():
    xtr, xte = _load(torchvision.datasets.FashionMNIST, data_dir("fashionmnist"))
    return xtr, xte  # (N, 28, 28) in [0,1]


def load_mnist():
    xtr, xte = _load(torchvision.datasets.MNIST, data_dir("mnist"))
    return xtr[:, None, :, :], xte[:, None, :, :]  # (N,1,28,28)
