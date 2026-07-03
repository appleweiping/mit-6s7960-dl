"""Common helpers: CPU thread config, seeding, and data loading.

All homeworks import from here so that runs are reproducible and CPU-bounded
(OMP_NUM_THREADS=3, torch.set_num_threads(3)) as required by the build spec.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

# Cap CPU threads for a modest, deterministic footprint on the build machine.
os.environ.setdefault("OMP_NUM_THREADS", "3")
torch.set_num_threads(3)


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """CPU-only build; return cpu explicitly so configs are unambiguous."""
    return torch.device("cpu")


def data_dir(*parts: str) -> str:
    """Return an absolute path under the repo-level, git-ignored ``data/`` dir."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, *parts)
    if parts:
        os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
