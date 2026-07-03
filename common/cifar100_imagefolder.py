"""Load CIFAR-100 (fine labels) from the fast.ai image-folder layout.

fast.ai ships CIFAR-100 as train/<superclass>/<fine_class>/*.png (100 fine
classes nested under 20 coarse superclasses). We map each fine-class folder to a
0..99 label (sorted alphabetically, a fixed deterministic ordering) and return
the same normalized CHW tensors the HW2 arch-comparison consumes.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

from common.utils import data_dir

MEAN = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32)
STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)


def _root():
    return os.path.join(data_dir("cifar100"), "cifar100")


def available() -> bool:
    return os.path.isdir(os.path.join(_root(), "train"))


def _fine_classes():
    """Deterministic 0..99 mapping from fine-class folder names."""
    train = os.path.join(_root(), "train")
    fines = set()
    for sup in os.listdir(train):
        sup_path = os.path.join(train, sup)
        if os.path.isdir(sup_path):
            for fine in os.listdir(sup_path):
                if os.path.isdir(os.path.join(sup_path, fine)):
                    fines.add(fine)
    return sorted(fines)


def _load_split(split, class_to_idx):
    root = os.path.join(_root(), split)
    imgs, labels = [], []
    for sup in sorted(os.listdir(root)):
        sup_path = os.path.join(root, sup)
        if not os.path.isdir(sup_path):
            continue
        for fine in sorted(os.listdir(sup_path)):
            fine_path = os.path.join(sup_path, fine)
            if not os.path.isdir(fine_path):
                continue
            label = class_to_idx[fine]
            for path in glob.glob(os.path.join(fine_path, "*.png")):
                imgs.append(np.asarray(Image.open(path).convert("RGB"),
                                       dtype=np.uint8))
                labels.append(label)
    x = np.stack(imgs).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(0, 3, 1, 2).copy()
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def load_images(cache=True):
    npz = os.path.join(_root(), "cifar100_cache.npz")
    if cache and os.path.exists(npz):
        d = np.load(npz)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    fines = _fine_classes()
    class_to_idx = {c: i for i, c in enumerate(fines)}
    xtr, ytr = _load_split("train", class_to_idx)
    xte, yte = _load_split("test", class_to_idx)
    if cache:
        np.savez(npz, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte
