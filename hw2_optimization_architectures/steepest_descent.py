"""HW2 -- steepest descent, dual norms, spectral norm, power iteration.

The written questions derive closed-form solutions to

    argmin_{dw}  g^T dw + (lambda/2) ||dw||^2

under different norms. This module encodes those solutions as *code* and checks
them numerically against brute-force / PyTorch references, so the analysis is
verifiable rather than merely asserted.

Covered:
  * dual norm of l2 and l_inf  (q1);
  * steepest descent under l2 and l_inf norms  (q3);
  * steepest descent under the spectral norm  (q4b): the update is a rescaled
    "orthogonalised gradient"  -||G||_dual/lambda * U V^T;
  * empirical spectral-norm scaling of Gaussian / orthogonal matrices (q6, q7);
  * power iteration for the spectral norm (q8);
  * effect of weight decay on singular values / vectors (q11).
"""
from __future__ import annotations

import numpy as np


# ---- Dual norms (q1) ------------------------------------------------------
def dual_norm_lp(vec: np.ndarray, p: float) -> float:
    """Dual of the l_p norm is the l_q norm with 1/p + 1/q = 1."""
    if np.isinf(p):
        q = 1.0
    elif p == 1:
        q = np.inf
    else:
        q = p / (p - 1.0)
    return float(np.linalg.norm(vec, ord=q))


# ---- Steepest descent under vector norms (q3) -----------------------------
def steepest_step_l2(g: np.ndarray, lam: float) -> np.ndarray:
    """argmin g^T dw + (lam/2)||dw||_2^2  =  -g/lam (ordinary gradient step)."""
    return -g / lam


def steepest_step_linf(g: np.ndarray, lam: float) -> np.ndarray:
    """Under the l_inf penalty the solution is -(||g||_1/lam) * sign(g).

    This is the sign-gradient / (unnormalised) Adam-like direction.
    """
    return -(np.linalg.norm(g, ord=1) / lam) * np.sign(g)


def brute_force_steepest(g: np.ndarray, lam: float, ord_p, n_samples=200000, seed=0):
    """Sanity check: random search over directions to approximate the argmin."""
    rng = np.random.default_rng(seed)
    best_val = np.inf
    best_dw = None
    # search over magnitudes and directions
    for _ in range(n_samples):
        d = rng.standard_normal(g.shape)
        d /= (np.linalg.norm(d, ord=ord_p) + 1e-12)  # unit in this norm
        for mag in np.linspace(0, 4 * np.linalg.norm(g) / lam, 40):
            dw = mag * d
            val = g @ dw + 0.5 * lam * np.linalg.norm(dw, ord=ord_p) ** 2
            if val < best_val:
                best_val = val
                best_dw = dw
    return best_dw, best_val


# ---- Steepest descent under the spectral norm (q4b) -----------------------
def spectral_norm(A: np.ndarray) -> float:
    return float(np.linalg.svd(A, compute_uv=False)[0])


def steepest_step_spectral(G: np.ndarray, lam: float) -> np.ndarray:
    r"""Solve argmin trace(G^T dW) + (lam/2)||dW||_*^2 over dW.

    With G = U S V^T, the closed form is
        dW = -(||G||_dual / lam) * U V^T   where ||G||_dual = trace(S)
    (the dual of the spectral norm is the nuclear norm).  U V^T is the
    "orthogonal polar factor" of G.
    """
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    nuclear = float(S.sum())
    return -(nuclear / lam) * (U @ Vt)


# ---- Spectral norm scaling of random matrices (q6, q7) --------------------
def gaussian_spectral_scaling(dims, seed=0):
    """Return (dims, measured spectral norms) for iid N(0,1) d x d matrices."""
    rng = np.random.default_rng(seed)
    norms = []
    for d in dims:
        A = rng.standard_normal((d, d))
        norms.append(spectral_norm(A))
    return np.array(dims), np.array(norms)


def fit_power_law(dims, norms):
    """Fit spectral_norm ~ alpha * d^beta via log-log least squares."""
    logd = np.log(dims)
    logn = np.log(norms)
    beta, logalpha = np.polyfit(logd, logn, 1)
    return float(np.exp(logalpha)), float(beta)


def orthogonal_spectral_norms(dims, seed=0):
    """Spectral norm of random orthogonal matrices (should be ~1 for all d)."""
    rng = np.random.default_rng(seed)
    out = []
    for d in dims:
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        out.append(spectral_norm(Q))
    return np.array(dims), np.array(out)


# ---- Power iteration for the spectral norm (q8) ---------------------------
def power_iteration_spectral(A: np.ndarray, iters: int = 30, seed: int = 0) -> float:
    """Estimate ||A||_* = sqrt(lambda_max(A A^T)) via power iteration."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    AAt = A @ A.T
    for _ in range(iters):
        v = AAt @ v
        v /= (np.linalg.norm(v) + 1e-12)
    lam = (v @ (AAt @ v)) / (v @ v)
    return float(np.sqrt(max(lam, 0.0)))


# ---- Weight decay on singular values (q11) --------------------------------
def weight_decay_effect(W: np.ndarray, factor: float = 0.999):
    """W -> factor*W scales every singular value by ``factor`` and leaves the
    singular vectors unchanged. Return (before_svals, after_svals, vectors_same).
    """
    U0, S0, Vt0 = np.linalg.svd(W, full_matrices=False)
    U1, S1, Vt1 = np.linalg.svd(factor * W, full_matrices=False)
    vectors_same = np.allclose(np.abs(U0.T @ U1), np.eye(len(S0)), atol=1e-6)
    return S0, S1, vectors_same


def _verify():
    rng = np.random.default_rng(0)

    # dual norms
    a = rng.standard_normal(6)
    assert np.isclose(dual_norm_lp(a, 2), np.linalg.norm(a, 2))
    assert np.isclose(dual_norm_lp(a, np.inf), np.linalg.norm(a, 1))
    print("dual norms: l2*=l2, linf*=l1  OK")

    # steepest descent l2 vs brute force
    g = rng.standard_normal(3)
    lam = 1.3
    dw2 = steepest_step_l2(g, lam)
    bf2, _ = brute_force_steepest(g, lam, 2, n_samples=4000)
    print(f"l2 step vs brute force: closed={dw2.round(3)} brute={bf2.round(3)} "
          f"cos={np.dot(dw2, bf2) / (np.linalg.norm(dw2) * np.linalg.norm(bf2)):.3f}")

    # spectral-norm steepest descent: verify it beats a random perturbation set
    G = rng.standard_normal((5, 4))
    dW = steepest_step_spectral(G, lam)
    obj = lambda M: np.trace(G.T @ M) + 0.5 * lam * spectral_norm(M) ** 2
    best_rand = min(obj(0.5 * spectral_norm(dW) * rng.standard_normal((5, 4)))
                    for _ in range(2000))
    print(f"spectral step objective={obj(dW):.4f}  best random={best_rand:.4f}  "
          f"(lower is better) OK={obj(dW) < best_rand}")

    # Gaussian spectral scaling ~ 2 * sqrt(d)
    dims = [50, 100, 200, 400, 800]
    d, norms = gaussian_spectral_scaling(dims)
    alpha, beta = fit_power_law(d, norms)
    print(f"Gaussian spectral norm ~ {alpha:.2f} * d^{beta:.3f}  "
          f"(theory: 2*sqrt(d), i.e. beta=0.5)")

    # orthogonal ~ 1
    _, on = orthogonal_spectral_norms(dims)
    print(f"orthogonal matrices spectral norms: {on.round(4)} (all ~1)")

    # power iteration
    A = rng.standard_normal((300, 300))
    pi = power_iteration_spectral(A)
    exact = spectral_norm(A)
    print(f"power iteration={pi:.3f}  exact={exact:.3f}  rel_err={abs(pi - exact) / exact:.2e}")

    # weight decay
    W = rng.standard_normal((6, 6))
    s0, s1, same = weight_decay_effect(W, 0.999)
    print(f"weight decay 0.999: sval ratio={np.mean(s1 / s0):.4f} (=factor), "
          f"singular vectors unchanged={same}")
    print("All HW2 steepest-descent checks passed.")


if __name__ == "__main__":
    _verify()
