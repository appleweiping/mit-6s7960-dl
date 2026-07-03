"""CIFAR-10 loading for HW1, returned as flat numpy arrays for the scratch MLP.

Uses torchvision only to fetch/parse the official CIFAR-10 files (downloaded at
runtime into the git-ignored ``data/`` dir), then hands back numpy tensors so the
from-scratch modules in ``modules.py`` can train without touching autograd.
"""
from __future__ import annotations

import numpy as np
import torchvision

from common.utils import data_dir


CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _to_numpy(dataset) -> tuple[np.ndarray, np.ndarray]:
    # dataset.data: uint8 (N,32,32,3); dataset.targets: list[int]
    x = np.asarray(dataset.data, dtype=np.float32) / 255.0
    x = (x - CIFAR10_MEAN) / CIFAR10_STD
    y = np.asarray(dataset.targets, dtype=np.int64)
    return x, y


def load_cifar10_flat(root: str | None = None):
    """Return (Xtr, ytr, Xte, yte) with images flattened to 3072-dim vectors.

    Prefers the fast.ai image-folder distribution (reliable CDN) if present,
    else falls back to the torchvision mirror download.
    """
    from common import cifar_imagefolder
    if cifar_imagefolder.available():
        return cifar_imagefolder.load_flat()
    root = root or data_dir("cifar10")
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    xtr, ytr = _to_numpy(train)
    xte, yte = _to_numpy(test)
    xtr = xtr.reshape(xtr.shape[0], -1)
    xte = xte.reshape(xte.shape[0], -1)
    return xtr, ytr, xte, yte


def load_cifar10_images(root: str | None = None):
    """Return CHW float tensors (for the CNN comparisons), normalized."""
    root = root or data_dir("cifar10")
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    xtr, ytr = _to_numpy(train)
    xte, yte = _to_numpy(test)
    xtr = xtr.transpose(0, 3, 1, 2).copy()
    xte = xte.transpose(0, 3, 1, 2).copy()
    return xtr, ytr, xte, yte
