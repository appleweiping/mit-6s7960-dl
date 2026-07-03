"""CIFAR-100 loader for HW2 (downloaded at runtime, git-ignored)."""
from __future__ import annotations

import numpy as np
import torchvision

from common.utils import data_dir

MEAN = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32)
STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)


def load_cifar100(root: str | None = None):
    from common import cifar100_imagefolder
    if cifar100_imagefolder.available():
        return cifar100_imagefolder.load_images()
    root = root or data_dir("cifar100")
    tr = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    te = torchvision.datasets.CIFAR100(root=root, train=False, download=True)

    def prep(ds):
        x = np.asarray(ds.data, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        x = x.transpose(0, 3, 1, 2).copy()
        y = np.asarray(ds.targets, dtype=np.int64)
        return x, y

    xtr, ytr = prep(tr)
    xte, yte = prep(te)
    return xtr, ytr, xte, yte
