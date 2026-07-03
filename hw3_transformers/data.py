"""Datasets for HW3: CIFAR-10 (for ViT) and tiny-shakespeare (for DialogueGPT)."""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import torchvision

from common.utils import data_dir

_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def load_cifar10_images(root: str | None = None):
    root = root or data_dir("cifar10")
    tr = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    def prep(ds):
        x = np.asarray(ds.data, dtype=np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = x.transpose(0, 3, 1, 2).copy()
        y = np.asarray(ds.targets, dtype=np.int64)
        return x, y

    xtr, ytr = prep(tr)
    xte, yte = prep(te)
    return xtr, ytr, xte, yte


def load_tiny_shakespeare() -> str:
    """Download the tiny-shakespeare corpus (a real, standard, MIT-friendly set)."""
    path = data_dir("shakespeare", "input.txt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def shakespeare_lines(min_len=20, max_lines=None):
    """Return non-empty lines of dialogue as individual training examples."""
    text = load_tiny_shakespeare()
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_len]
    if max_lines:
        lines = lines[:max_lines]
    return lines
