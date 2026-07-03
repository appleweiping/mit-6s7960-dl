"""HW1 -- constructive ReLU-network results (Approximation & Backprop sections).

These are the *coding-verifiable* parts of the written questions:

* Logic-gate ReLU networks: explicit weights implementing OR and XOR decision
  boundaries (questions 7a, 7b).
* Backprop by hand: the closed-form dL/dW for the chained system
  y=Wx, u=ReLU(y), v=u+Wu, L=0.5||v||^2 (question 8), checked numerically.

Running this file prints the verification that the constructed gates realise the
required boolean decision boundaries and that the hand-derived gradient matches
finite differences.
"""
from __future__ import annotations

import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


# --- Question 7a: OR gate --------------------------------------------------
# f(x) > 0  <=>  x1 > 0 OR x2 > 0.
# A single hidden layer that fires if either coordinate is positive, followed by
# a readout that is positive iff at least one branch fired.
OR_W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
OR_b1 = np.array([0.0, 0.0])
OR_W2 = np.array([[1.0, 1.0]])
OR_b2 = np.array([-1e-9])  # strictly > 0 only when some ReLU branch is active


def or_gate(x: np.ndarray) -> float:
    h = relu(OR_W1 @ x + OR_b1)
    return float((OR_W2 @ h + OR_b2)[0])


# --- Question 7b: XOR gate -------------------------------------------------
# f(x) > 0  <=>  (x1<0 AND x2>0) OR (x1>0 AND x2<0).
#
# Idea: an AND of two open half-planes {a>0} and {b>0} can be written with two
# ReLUs as  AND(a, b) = ReLU(a) - ReLU(a - b)  when we also gate by b>0.
# A cleaner, exact construction uses the identity
#     min(a, b) = a - ReLU(a - b),        for the "both large" test,
# and  branch = ReLU( min(a, b) )  is > 0  iff  a>0 AND b>0.
# Each opposite-sign branch is then a 2-layer sub-network; summing the two
# branches gives a 3-layer network of width <= 4 overall.
def _and_positive(a: float, b: float) -> float:
    """> 0 iff a > 0 AND b > 0 (returns min(a, b) clamped at 0)."""
    m = a - relu(a - b)          # = min(a, b), one ReLU
    return relu(m)               # keep only the positive part


def xor_gate(x: np.ndarray) -> float:
    x1, x2 = float(x[0]), float(x[1])
    # Half-plane scores (positive = inside the half-plane).
    branch_a = _and_positive(-x1, x2)   # x1 < 0 AND x2 > 0
    branch_b = _and_positive(x1, -x2)   # x1 > 0 AND x2 < 0
    # Exactly one branch can be positive at a time, so a plain sum works.
    return float(branch_a + branch_b)


# --- Question 8: hand-derived backprop -------------------------------------
def system_loss(W: np.ndarray, x: np.ndarray):
    y = W @ x
    u = relu(y)
    v = u + W @ u
    L = 0.5 * float(v @ v)
    return L, (y, u, v)


def system_grad_analytic(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""dL/dW = v (x) u  +  diag(Theta(y)) (I + W^T) v  (x) x  (outer products)."""
    _, (y, u, v) = system_loss(W, x)
    theta = (y >= 0).astype(float)
    term1 = np.outer(v, u)
    inner = np.diag(theta) @ (np.eye(W.shape[0]) + W.T) @ v
    term2 = np.outer(inner, x)
    return term1 + term2


def system_grad_numeric(W: np.ndarray, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += eps
            Wm = W.copy(); Wm[i, j] -= eps
            g[i, j] = (system_loss(Wp, x)[0] - system_loss(Wm, x)[0]) / (2 * eps)
    return g


def _verify() -> None:
    rng = np.random.default_rng(0)

    # OR gate truth over a grid of non-boolean inputs.
    ok_or = True
    for _ in range(2000):
        x = rng.uniform(-3, 3, size=2)
        expected = (x[0] > 0) or (x[1] > 0)
        got = or_gate(x) > 0
        ok_or &= (expected == got)
    print(f"OR gate matches (x1>0 OR x2>0) on 2000 random points: {ok_or}")

    # XOR gate truth (avoid the measure-zero axes where signs are ill-defined).
    ok_xor = True
    for _ in range(2000):
        x = rng.uniform(-3, 3, size=2)
        if abs(x[0]) < 1e-3 or abs(x[1]) < 1e-3:
            continue
        expected = ((x[0] < 0) and (x[1] > 0)) or ((x[0] > 0) and (x[1] < 0))
        got = xor_gate(x) > 0
        ok_xor &= (expected == got)
    print(f"XOR gate matches opposite-sign region on random points: {ok_xor}")

    # Backprop check.
    d = 4
    W = rng.standard_normal((d, d))
    x = rng.standard_normal(d)
    ga = system_grad_analytic(W, x)
    gn = system_grad_numeric(W, x)
    rel = np.abs(ga - gn).max() / (np.abs(gn).max() + 1e-12)
    print(f"Backprop dL/dW  max rel error vs finite diff: {rel:.2e}")
    assert ok_or and ok_xor and rel < 1e-5, "verification failed"
    print("All HW1 constructive/backprop checks passed.")


if __name__ == "__main__":
    _verify()
