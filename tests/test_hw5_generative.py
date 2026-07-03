"""Shape / correctness tests for the HW5 VAE and diffusion implementations."""
import torch

from hw5_generative_models.vae import VAE, elbo_loss, plot_latents
from hw5_generative_models.diffusion import (
    DiffusionSchedule, UNet, diffusion_loss, sample,
)


def test_vae_forward_and_elbo():
    model = VAE(latent=8)
    x = torch.rand(5, 784)
    recon, mu, logvar = model(x)
    assert recon.shape == (5, 784)
    assert mu.shape == (5, 8) and logvar.shape == (5, 8)
    loss, rec, kl = elbo_loss(recon, x, mu, logvar)
    assert loss.item() > 0 and kl.item() >= 0


def test_vae_kl_zero_at_standard_normal():
    # KL(N(0,I)||N(0,I)) == 0 when mu=0, logvar=0.
    x = torch.rand(4, 784)
    recon = torch.rand(4, 784).clamp(1e-4, 1 - 1e-4)
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    _, _, kl = elbo_loss(recon, x, mu, logvar)
    assert abs(kl.item()) < 1e-5


def test_plot_latents_shape():
    model = VAE(latent=8)
    canvas = plot_latents(model, 0, 1, grid=5, span=2.0)
    assert canvas.shape == (5 * 28, 5 * 28)


def test_diffusion_forward_closed_form():
    sched = DiffusionSchedule(T=50)
    x0 = torch.rand(3, 1, 28, 28)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 25, 49])
    xt = sched.q_sample(x0, t, noise)
    assert xt.shape == x0.shape
    # at t=0, alpha_bar ~ 1, so x_t ~= x_0
    xt0 = sched.q_sample(x0[:1], torch.tensor([0]), noise[:1])
    assert torch.allclose(xt0, x0[:1], atol=0.1)


def test_unet_and_loss_and_sample():
    sched = DiffusionSchedule(T=20)
    model = UNet(channels=1, base=8)
    x0 = torch.rand(2, 1, 28, 28)
    loss = diffusion_loss(model, sched, x0)
    assert loss.item() > 0
    imgs, caps = sample(model, sched, n=2, capture_ts=[10, 0])
    assert imgs.shape == (2, 1, 28, 28)
    assert 10 in caps and 0 in caps
