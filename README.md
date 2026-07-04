# MIT 6.7960 Deep Learning — Problem Set Solutions

> From-scratch implementations and analyses of the core neural-network building
> blocks — an independent, from-scratch implementation of
> **6.7960 — Deep Learning** (MIT, Fall 2024), part of a
> [csdiy.wiki](https://csdiy.wiki/) full-catalog build.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![language](https://img.shields.io/badge/python-informational)
![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

MIT 6.7960 builds up deep learning from first principles: network architectures
(MLPs, CNNs, GNNs, transformers), optimisation geometry, representation learning,
and generative models. This repo implements every coding portion of the five
problem sets **from scratch** — hand-derived backprop, an attention/transformer
stack, contrastive and reconstruction encoders, a VAE and a DDPM — and trains
each on the assignment's real datasets (CIFAR-10/100, tiny-shakespeare,
FashionMNIST, MNIST). The written/theory questions are answered in
[`notes/solution_notes.md`](notes/solution_notes.md). Everything runs on **CPU
only** and is verified with a pytest suite (29 tests) plus real training runs
whose measured outputs are saved under [`results/`](results/).

## Results (measured on CPU, `OMP_NUM_THREADS=3`, torch 2.12 CPU)

| Pset | What it does | Result (measured) |
|---|---|---|
| **HW1** scratch-MLP on CIFAR-10 | hand-derived Linear/ReLU/CE backward | no-aug **train 0.987 / val 0.535** (overfits); random-crop aug **train 0.438 / val 0.432** (gap closed) |
| **HW1** ReLU logic gates + backprop | OR/XOR nets, `∂L/∂W` closed form | gates correct on 2000 random points; grad matches finite-diff to `1.3e-10` |
| **HW2** steepest descent | dual norms, spectral-norm step, power iter | Gaussian spectral norm `≈1.8·d^0.52` (theory `2√d`); orthogonal norm `=1`; power-iter rel-err `7e-3` |
| **HW2** LR transfer across width | spectral vs naive update | spectral: small-width-best LR still works at 8× width (loss 3× best); **naive diverges (NaN)** — does not transfer |
| **HW2** MLP vs CNN (CIFAR-100) | inductive bias | MLP d3 **0.134** ≈ MLP d7 **0.127** < CNN **0.181** (20k subset); on full data CNN reached **0.41** vs MLP **0.23** — deeper MLP does not help, CNN does |
| **HW2** GNN aggregation | mean/max message passing | mean exact; soft-max within `5e-2` of true max |
| **HW3** Transformer + ViT (CIFAR-10) | attention/multihead/ViT from scratch | ViT **val acc 0.62** (assignment target: >0.50) |
| **HW3** DialogueGPT (tiny-shakespeare) | GPT LM on the from-scratch transformer | next-token loss **6.55 → 4.29**; generates grammatical Shakespeare-style text |
| **HW4** representation learning | autoencoder vs contrastive, k-NN probe | recolor-aug encoder: shape acc **0.95**, colour acc **0.25 (= chance)** → colour-invariant, as theory predicts |
| **HW5** VAE (FashionMNIST) | ELBO + reparameterisation | final **−ELBO 240.9** (recon 224.9, KL 16.0); latent traversal figure |
| **HW5** DDPM diffusion (MNIST) | ε-parameterised, time-conditioned U-Net | final noise-pred loss **0.0497**; samples + reverse-process figure |

Figures and metric JSONs live under [`results/`](results/) (e.g.
`results/hw1/cifar10_curves.png`, `results/hw3/vit_attention.png`,
`results/hw4/nn_grid.png`, `results/hw5/diffusion_reverse.png`).

## Implemented assignments

- [x] **HW1 — Foundations** — ReLU-net theory; from-scratch Linear/ReLU/CrossEntropy
  forward+backward (no autograd); constructive OR/XOR gates; hand-derived `∂L/∂W`;
  CIFAR-10 training + data augmentation.
- [x] **HW2 — Optimization & architectures** — steepest descent under `ℓ2/ℓ∞`/spectral
  norms; spectral-norm scaling, power iteration, weight decay; learning-rate transfer
  across width; MLP-vs-CNN inductive bias; MP-GNN mean/max aggregation.
- [x] **HW3 — Transformers** — self-attention, multi-head, FFN, residual blocks,
  full Transformer; Vision Transformer (patch embed, class token, attention maps);
  DialogueGPT (word tokenizer, causal mask, next-token loss, greedy generation).
- [x] **HW4 — Representation learning** — colored-shapes dataset; autoencoder
  (reconstruction) vs contrastive (InfoNCE) encoders; augmentation-defined
  invariances quantified by k-NN factor probing.
- [x] **HW5 — Generative models** — Variational Autoencoder (ELBO, reparameterisation,
  latent traversal); Denoising Diffusion Probabilistic Model (forward closed form,
  ε-loss derivation, ancestral sampling).

## Project structure

```
mit-6s7960-dl/
├── common/                       # shared utils, robust CIFAR image-folder loaders
├── hw1_foundations/              # scratch modules, ReLU gates, CIFAR-10 trainer
├── hw2_optimization_architectures/  # steepest descent, LR transfer, arch compare, GNN
├── hw3_transformers/             # transformer, ViT, DialogueGPT + trainers
├── hw4_representation_learning/  # shapes dataset, AE vs contrastive, k-NN probe
├── hw5_generative_models/        # VAE + DDPM diffusion
├── notes/solution_notes.md       # written/theory answers (HW1–HW5)
├── tests/                        # 29 pytest checks (grad-checks vs autograd, etc.)
├── results/                      # measured figures + metric JSONs
└── requirements.txt
```

## How to run

```bash
# Python repos use the shared csdiy env (Python 3.11):
#   D:\Project\_csdiy\.venv-ml\Scripts\python.exe
python -m pip install -r requirements.txt

# Verify all from-scratch backward passes / constructions against autograd:
python -m pytest -q

# HW1: hand-derived-backprop MLP on CIFAR-10 (learning curves + augmentation)
python -m hw1_foundations.train_cifar10 --epochs 30
python hw1_foundations/relu_networks.py            # OR/XOR gates + backprop check

# HW2: optimisation-geometry checks, LR transfer, MLP-vs-CNN, GNN
python hw2_optimization_architectures/steepest_descent.py
python -m hw2_optimization_architectures.lr_transfer
python -m hw2_optimization_architectures.arch_compare --epochs 20
python hw2_optimization_architectures/gnn_aggregation.py

# HW3: ViT on CIFAR-10, DialogueGPT on tiny-shakespeare
python -m hw3_transformers.train_vit --epochs 8
python -m hw3_transformers.train_gpt --epochs 8

# HW4: reconstruction vs contrastive representations + k-NN probe
python -m hw4_representation_learning.train_repr

# HW5: VAE (FashionMNIST) and DDPM diffusion (MNIST)
python -m hw5_generative_models.train_vae --epochs 15
python -m hw5_generative_models.train_diffusion --epochs 8
```

Datasets are downloaded at runtime into a git-ignored `data/` directory
(CIFAR-10/100 fall back to the fast.ai image-folder mirror when the torchvision
mirror is unreachable), so no course data is redistributed here.

## Verification

- **Grad-checks:** the HW1 from-scratch `Linear`/`ReLU`/`CrossEntropyLoss`
  backward passes are checked against PyTorch autograd; the HW1 question-8 matrix
  gradient and the ReLU logic gates are checked by finite differences / exhaustive
  sampling (`tests/test_hw1_gradients.py`).
- **Attention mask:** the transformer's causal/masked attention is verified to put
  zero weight on masked positions (`tests/test_hw3_transformer.py`).
- **Closed forms:** HW2 dual norms, spectral-norm steepest-descent step (polar
  factor), power iteration, and weight-decay singular-value scaling are all checked
  against SVD references (`tests/test_hw2_optimization.py`).
- **VAE/diffusion:** ELBO KL vanishes at the standard normal; the forward diffusion
  closed form matches at `t=0`; sampling runs end-to-end (`tests/test_hw5_generative.py`).
- **Real runs:** every trainer above was executed and its measured numbers/figures
  saved under `results/` (see the table).

Run `python -m pytest -q` — all 29 tests pass.

## Tech stack

Python 3.11, NumPy (from-scratch HW1 autodiff), PyTorch 2.12 (CPU),
torchvision (dataset parsing), matplotlib. CPU-only, `OMP_NUM_THREADS=3`,
`torch.set_num_threads(3)`.

## Key ideas / what I learned

- Deriving and coding backprop by hand (per-layer Jacobians, the `v⊗u + …` matrix
  gradient) and checking it against autograd.
- Optimisation as geometry: steepest descent under different norms recovers SGD,
  sign-GD, and the spectral (polar-factor) update; spectral scaling is what makes
  learning rates transfer across width.
- Inductive bias: CNNs beat MLPs on images and depth alone doesn't fix an MLP;
  message-passing GNNs' expressive limits (e.g. can't count triangles / see chirality).
- Attention from scratch, and how the same transformer serves both images (ViT) and
  language (GPT) with the right tokenisation/positional encoding.
- Representation learning: reconstruction preserves invertibility but not semantics,
  while contrastive learning makes representations invariant to exactly the augmented
  factors — measurable via k-NN probes.
- Generative modelling: the ELBO and the DDPM ε-loss both reduce to simple training
  objectives whose derivations are worked out in the notes.

## Credits & license

Based on the assignments of **6.7960 Deep Learning** (Fall 2024) by the MIT EECS
department (materials on [MIT OpenCourseWare](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/),
CC BY-NC-SA). This repository is an independent educational reimplementation; all
course materials, datasets, and specifications belong to their original authors.
Original code in this repo is released under the [MIT License](LICENSE).
