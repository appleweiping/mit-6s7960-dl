"""Verify HW1 from-scratch backward passes against PyTorch autograd."""
import numpy as np
import torch

from hw1_foundations.modules import Linear, ReLU, CrossEntropyLoss, MLP
from hw1_foundations.relu_networks import (
    or_gate, xor_gate, system_grad_analytic, system_grad_numeric,
)


def test_linear_backward_matches_autograd():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 4))
    lin = Linear(4, 3, seed=1)
    out = lin.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = lin.backward(dout)

    xt = torch.tensor(x, requires_grad=True)
    Wt = torch.tensor(lin.W, requires_grad=True)
    bt = torch.tensor(lin.b, requires_grad=True)
    outt = xt @ Wt.T + bt
    outt.backward(torch.tensor(dout))
    assert np.allclose(dx, xt.grad.numpy(), atol=1e-8)
    assert np.allclose(lin.dW, Wt.grad.numpy(), atol=1e-8)
    assert np.allclose(lin.db, bt.grad.numpy(), atol=1e-8)


def test_relu_backward_matches_autograd():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 7))
    relu = ReLU()
    out = relu.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = relu.backward(dout)

    xt = torch.tensor(x, requires_grad=True)
    torch.relu(xt).backward(torch.tensor(dout))
    assert np.allclose(dx, xt.grad.numpy(), atol=1e-8)


def test_cross_entropy_backward_matches_autograd():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((8, 10))
    targets = rng.integers(0, 10, size=8)
    ce = CrossEntropyLoss()
    loss = ce.forward(logits, targets)
    dlogits = ce.backward()

    lt = torch.tensor(logits, requires_grad=True)
    loss_t = torch.nn.functional.cross_entropy(lt, torch.tensor(targets))
    loss_t.backward()
    assert abs(loss - loss_t.item()) < 1e-8
    assert np.allclose(dlogits, lt.grad.numpy(), atol=1e-8)


def test_mlp_full_backward_matches_autograd():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 12))
    targets = rng.integers(0, 5, size=4)
    mlp = MLP([12, 8, 5], seed=0)
    logits = mlp.forward(x)
    mlp.loss(logits, targets)
    mlp.backward()

    # Rebuild the same net in torch and compare grads of the first Linear.
    lin0 = mlp.layers[0]
    Wt = torch.tensor(lin0.W, requires_grad=True)
    bt = torch.tensor(lin0.b, requires_grad=True)
    lin1 = mlp.layers[2]
    W1t = torch.tensor(lin1.W, requires_grad=True)
    b1t = torch.tensor(lin1.b, requires_grad=True)
    xt = torch.tensor(x)
    h = torch.relu(xt @ Wt.T + bt)
    out = h @ W1t.T + b1t
    loss_t = torch.nn.functional.cross_entropy(out, torch.tensor(targets))
    loss_t.backward()
    assert np.allclose(lin0.dW, Wt.grad.numpy(), atol=1e-7)
    assert np.allclose(lin1.dW, W1t.grad.numpy(), atol=1e-7)


def test_logic_gates():
    rng = np.random.default_rng(4)
    for _ in range(500):
        x = rng.uniform(-3, 3, size=2)
        assert (or_gate(x) > 0) == (x[0] > 0 or x[1] > 0)
        if abs(x[0]) > 1e-3 and abs(x[1]) > 1e-3:
            exp = (x[0] < 0 and x[1] > 0) or (x[0] > 0 and x[1] < 0)
            assert (xor_gate(x) > 0) == exp


def test_hand_backprop_system():
    rng = np.random.default_rng(5)
    W = rng.standard_normal((4, 4))
    x = rng.standard_normal(4)
    ga = system_grad_analytic(W, x)
    gn = system_grad_numeric(W, x)
    assert np.abs(ga - gn).max() < 1e-5
