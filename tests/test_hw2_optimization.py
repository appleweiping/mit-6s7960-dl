"""Tests for HW2 steepest-descent / spectral-norm and GNN aggregation."""
import numpy as np

from hw2_optimization_architectures.steepest_descent import (
    dual_norm_lp, steepest_step_l2, steepest_step_linf, steepest_step_spectral,
    spectral_norm, gaussian_spectral_scaling, fit_power_law,
    orthogonal_spectral_norms, power_iteration_spectral, weight_decay_effect,
)
from hw2_optimization_architectures.gnn_aggregation import (
    aggregate_mean, MaxAggregator, MPLayer,
)


def test_dual_norms():
    a = np.random.default_rng(0).standard_normal(6)
    assert np.isclose(dual_norm_lp(a, 2), np.linalg.norm(a, 2))
    assert np.isclose(dual_norm_lp(a, np.inf), np.linalg.norm(a, 1))
    assert np.isclose(dual_norm_lp(a, 1), np.linalg.norm(a, np.inf))


def test_steepest_l2_is_gradient_step():
    g = np.array([1.0, -2.0, 3.0])
    assert np.allclose(steepest_step_l2(g, 2.0), -g / 2.0)


def test_steepest_linf_is_sign_scaled():
    g = np.array([1.0, -2.0, 3.0])
    step = steepest_step_linf(g, 1.0)
    assert np.allclose(np.sign(step), -np.sign(g))


def test_spectral_step_uses_polar_factor():
    rng = np.random.default_rng(1)
    G = rng.standard_normal((4, 3))
    dW = steepest_step_spectral(G, 1.0)
    # -dW should be a positive multiple of the orthogonal polar factor U V^T
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    polar = U @ Vt
    # cosine between -dW and polar (flattened) ~ 1
    a = (-dW).ravel(); b = polar.ravel()
    assert np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.999


def test_gaussian_spectral_scaling_beta_half():
    dims = [64, 128, 256, 512]
    d, norms = gaussian_spectral_scaling(dims, seed=0)
    _, beta = fit_power_law(d, norms)
    assert 0.4 < beta < 0.6   # theory: sqrt(d) => beta ~ 0.5


def test_orthogonal_spectral_norm_is_one():
    _, on = orthogonal_spectral_norms([50, 100, 200])
    assert np.allclose(on, 1.0, atol=1e-4)


def test_power_iteration_matches_svd():
    A = np.random.default_rng(2).standard_normal((200, 200))
    pi = power_iteration_spectral(A, iters=40)
    exact = spectral_norm(A)
    assert abs(pi - exact) / exact < 0.02


def test_weight_decay_scales_singular_values():
    W = np.random.default_rng(3).standard_normal((5, 5))
    s0, s1, same = weight_decay_effect(W, 0.9)
    assert np.allclose(s1 / s0, 0.9, atol=1e-6)
    assert same


def test_mean_aggregation_exact():
    rng = np.random.default_rng(4)
    nbrs = [rng.standard_normal(4) for _ in range(5)]
    assert np.allclose(aggregate_mean(nbrs), np.mean(nbrs, axis=0), atol=1e-6)


def test_max_aggregation_approx():
    rng = np.random.default_rng(5)
    agg = MaxAggregator(p=60.0)
    nbrs = [rng.standard_normal(4) for _ in range(5)]
    assert np.allclose(agg(nbrs), np.max(nbrs, axis=0), atol=5e-2)


def test_mp_layer_single_neighbour():
    adj = np.array([[0, 1], [1, 0]])
    feats = np.random.default_rng(6).standard_normal((2, 3))
    out = MPLayer("mean").forward(adj, feats)
    assert np.allclose(out[0], feats[1], atol=1e-6)
