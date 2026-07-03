"""Load CIFAR-10 from the fast.ai image-folder layout (PNG per image).

The canonical torchvision CIFAR-10 mirror (cs.toronto.edu) is unreliable from
some networks, so as a robust fallback we read the fast.ai distribution, which
ships the identical 60k images as PNGs under train/<class>/ and test/<class>/.
This yields the same normalized numpy tensors the HW1/HW3 scripts consume.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

from common.utils import data_dir

# fast.ai folder uses these 10 class names (CIFAR-10 order by torchvision label).
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _root():
    return os.path.join(data_dir("cifar10"), "cifar10")


def available() -> bool:
    return os.path.isdir(os.path.join(_root(), "train"))


def _load_split(split):
    root = os.path.join(_root(), split)
    imgs, labels = [], []
    for label, cls in enumerate(CLASSES):
        for path in sorted(glob.glob(os.path.join(root, cls, "*.png"))):
            imgs.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
            labels.append(label)
    x = np.stack(imgs).astype(np.float32) / 255.0        # (N,32,32,3)
    x = (x - MEAN) / STD
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def load_flat(cache=True):
    """Flattened 3072-dim vectors (for the HW1 scratch MLP).

    Reuses the same npz cache as ``load_images`` so the (slow) per-PNG read only
    happens once across all homeworks.
    """
    x_tr, y_tr, x_te, y_te = load_images(cache=cache)
    return (x_tr.reshape(len(x_tr), -1), y_tr,
            x_te.reshape(len(x_te), -1), y_te)


def load_images(cache=True):
    """CHW normalized float tensors (for CNN / ViT)."""
    npz = os.path.join(_root(), "cifar10_cache.npz")
    if cache and os.path.exists(npz):
        d = np.load(npz)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    xtr, ytr = _load_split("train")
    xte, yte = _load_split("test")
    xtr = xtr.transpose(0, 3, 1, 2).copy()
    xte = xte.transpose(0, 3, 1, 2).copy()
    if cache:
        np.savez(npz, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte
