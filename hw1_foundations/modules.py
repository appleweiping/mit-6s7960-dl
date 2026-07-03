"""HW1 -- neural network layers with hand-derived forward and backward passes.

Implements the module classes the assignment asks for (Linear, ReLU,
CrossEntropyLoss) *without* using autograd. Each module exposes ``forward`` and
``backward``; the backward pass is derived analytically from the chain rule so
that a small MLP can be trained on CIFAR-10 with only these primitives.

The gradients here are verified against PyTorch autograd in
``tests/test_hw1_gradients.py``.
"""
from __future__ import annotations

import numpy as np


class Module:
    """Base class: a module caches whatever its backward pass needs."""

    def __init__(self) -> None:
        self.cache = {}

    def forward(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    def parameters(self):
        return []

    def grads(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    r"""Affine layer ``out = x W^T + b``.

    Forward:  out = x @ W.T + b, with x in R^{N x din}, W in R^{dout x din}.
    Backward (given dL/dout):
        dL/dx = dL/dout @ W
        dL/dW = dL/dout^T @ x
        dL/db = sum_n dL/dout[n]
    """

    def __init__(self, din: int, dout: int, seed: int | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        # Kaiming-style init for ReLU nets.
        self.W = rng.standard_normal((dout, din)) * np.sqrt(2.0 / din)
        self.b = np.zeros(dout)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x
        return x @ self.W.T + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache["x"]
        self.dW = dout.T @ x
        self.db = dout.sum(axis=0)
        return dout @ self.W

    def parameters(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU(Module):
    r"""Elementwise ReLU. Backward gates the gradient where the input was < 0."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["mask"] = x > 0
        return np.where(x > 0, x, 0.0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.cache["mask"]


class CrossEntropyLoss(Module):
    r"""Softmax + cross entropy over a batch.

    Forward returns the mean loss over the batch. Backward returns dL/dlogits.
    For a single example with softmax probabilities p and one-hot target y:
        dL/dlogits = p - y   (averaged over the batch).
    """

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        # numerically stable softmax
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)
        n = logits.shape[0]
        self.cache["probs"] = probs
        self.cache["targets"] = targets
        self.cache["n"] = n
        log_likelihood = -np.log(probs[np.arange(n), targets] + 1e-12)
        return float(log_likelihood.mean())

    def backward(self) -> np.ndarray:
        probs = self.cache["probs"].copy()
        targets = self.cache["targets"]
        n = self.cache["n"]
        probs[np.arange(n), targets] -= 1.0
        return probs / n


class MLP:
    """A small multi-layer perceptron built from the modules above."""

    def __init__(self, sizes, seed: int = 0) -> None:
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], seed=seed + i))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.loss_fn.forward(logits, targets)

    def backward(self) -> None:
        dout = self.loss_fn.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def step(self, lr: float) -> None:
        for layer in self.layers:
            for p, g in zip(layer.parameters(), layer.grads()):
                p -= lr * g

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
