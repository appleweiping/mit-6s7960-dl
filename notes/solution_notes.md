# Solution notes — written/theory portions

Concise, self-contained answers to the non-coding questions of MIT 6.7960
(Fall 2024) HW1–HW5. The coding portions are implemented in the `hwN_*/`
packages and verified with the tests and the runs recorded in `results/`.
These notes are our own derivations, phrased in our own words.

---

## HW1 — Foundations

**Shapes (Q1).** For a 2-layer ReLU net with hidden width `k` and
`W1 ∈ R^{k×1}`: `W2 ∈ R^{1×k}`, `b1 ∈ R^{k}`, `b2 ∈ R^{1}`.

**Depth expression (Q2).** `f = W3 ReLU(W2 ReLU(W1 x + b1) + b2) + b3`.

**Convexity (Q3).** For `l ≥ 2` a ReLU network is in general **neither** convex
nor concave in the input (a composition of convex ReLUs with sign-varying linear
maps loses both).

**Piecewise linearity & non-differentiable points (Q4).**
- The output is **piecewise-linear** (choice B), continuous everywhere.
- It is not differentiable at the kinks. A single width-`k` hidden layer can add
  up to `k` breakpoints. For `l = 2`, at most `k` non-differentiable points.
- The number of linear regions can grow **exponentially in depth** (choice D):
  each new layer can subdivide every existing region, so deep nets are
  exponentially more efficient at representing highly-oscillatory
  piecewise-linear functions than a shallow net of the same neuron budget.
- With a smooth `atanh` nonlinearity the network is differentiable everywhere
  (the composition of smooth functions is smooth), so there are no kinks.

**Smoothness constructions (Q5).** With width-2, no biases:
`f(x) = W2 ReLU(W1 x)`. Choosing `W1 = [1, -1]^T`, `W2 = [1, 1]` gives
`f(x) = ReLU(x) + ReLU(-x) = |x|`, which has one non-differentiable point; a
linear `f` is obtained e.g. with `W1=[1,1]^T`, `W2=[1,-1]` on `x≥0` etc. Two
kinks require the two hidden pre-activations to cross zero at two distinct `x`,
impossible without biases through the origin — so two kinks need biases (handled
by the constructive `relu_networks.py`).

**Logic gates (Q7).** Implemented and verified in
`hw1_foundations/relu_networks.py`:
- OR: single hidden layer `h = ReLU([x1; x2])`, readout `1·h1 + 1·h2`, positive
  iff `x1>0` or `x2>0`.
- XOR: two "opposite-sign" branches, each an AND of two half-planes built from
  `min(a,b) = a − ReLU(a−b)` then `ReLU(min)`; the sum of the two branches is
  positive exactly on the XOR region. ≤3 layers, width ≤4.

**Backprop (Q8).** For `y=Wx, u=ReLU(y), v=u+Wu, L=½‖v‖²` the hand-derived
gradient is `∂L/∂W = v ⊗ u + diag(Θ(y))(I + Wᵀ) v ⊗ x`, matching finite
differences to `~1e-10` (checked in code).

**CIFAR-10 (Q12–15).** Universal approximation says a big enough net *can* fit
the training set; we do not reach 100% train accuracy because of finite width,
finite optimisation budget, and (crucially) we care about **generalisation**,
not memorisation. Random-crop augmentation shrinks the train/val gap (a mild
regulariser), which the two learning curves in `results/hw1/` show.

---

## HW2 — Optimization & architectures

**Dual norms (Q1).** `‖·‖₂† = ‖·‖₂` (Euclidean is self-dual);
`‖·‖∞† = ‖·‖₁`. In general `ℓp† = ℓq` with `1/p + 1/q = 1`.

**Dual formulation (Q2).** Writing `Δw = r·t` with `‖t‖=1` and minimising over
magnitude `r` gives `Δw* = −(‖g‖†/λ)·argmax_{‖t‖=1} gᵀt`.

**Steepest descent (Q3).**
- Under `ℓ₂`: `Δw* = −g/λ` (ordinary gradient step).
- Under `ℓ∞`: `Δw* = −(‖g‖₁/λ)·sign(g)` (sign-gradient / Adam-like direction).

**Spectral norm (Q4).** With `G = UΣVᵀ`, the steepest-descent step under the
spectral norm is `ΔW* = −(tr Σ / λ)·U Vᵀ` — the gradient's **orthogonal polar
factor**, scaled by the nuclear norm (dual of the spectral norm). Verified in
`steepest_descent.py`.

**Random matrices (Q6–8).** The spectral norm of an iid `N(0,1)` `d×d` matrix
scales like `2√d` (our fit gives `≈1.8·d^0.52`, consistent with the Marchenko–
Pastur edge `2√d`). A random orthogonal matrix has spectral norm exactly 1 for
all `d`. Power iteration slightly **under-estimates** the spectral norm (it
lower-bounds via a Rayleigh quotient) but is far cheaper than a full SVD.

**LR transfer (Q9).** With semi-orthogonal init `Wk = √(dk/dk₋₁)·Mk` and the
spectral-scaled sign-gradient update, the best learning rate found at small
width also works at large width. A naive elementwise update does **not**
transfer. Demonstrated empirically in `lr_transfer.py`.

**Weight decay (Q11).** `W → 0.999·W` multiplies **every singular value** by
0.999 and leaves the **singular vectors unchanged** (shrinks all directions
uniformly). Verified numerically.

**Architecture (Q12–14).** A CNN beats both MLPs on CIFAR-100 because its
locality + weight-sharing inductive bias matches image statistics; making the
MLP deeper (d=3→7) does not close the gap and can hurt (harder to optimise, no
spatial prior). See `results/hw2/arch_compare.*`.

**GNN aggregation (Q15).**
- **Mean**: append a constant 1 to each message (`g(h)=[h;1]`), sum, then divide
  the feature part by the accumulated count inside `f`.
- **Max** (approx): `g(h)=exp(p·h)`, `f(s)=log(s)/p`; as `p→∞` this converges to
  the coordinatewise max (a smooth `L∞`).

**Power of MP-GNNs (Q16).** Node-count with identical features: solvable with
mean/generic (readout = sum of ones / mean-count trick); max cannot count.
Shortest-path/eccentricity: solvable with **max** aggregation (BFS-style
distance propagation). Triangle-counting: **not** solvable by 1-WL message
passing — two graphs with identical neighbourhood trees but different triangle
counts are indistinguishable.

**Chirality (Q17).** No. `generic_{f,g}` uses a **permutation-invariant sum**, so
it cannot distinguish a chiral molecule from its mirror image (mirror symmetry
preserves the multiset of neighbourhoods).

---

## HW3 — RNNs vs transformers

**RNN unrolling (Q1a–c).** With `φh = id`, `bh=by=0`, `h0=0`:
`h_t = Σ_{s=1}^{t} Wh^{t−s} Wx x_s`, and `y_t = φy(Wy h_t)`. The contribution of
the first input `x1` to `y_T` is scaled by `Wh^{T−1}`; for `‖Wh‖<1` it decays
geometrically — the **vanishing-gradient / long-range** problem.

**Complexity (Q1e).** RNN forward pass: `O(T)` sequential steps (cannot be
parallelised over time). Self-attention: `O(T²)` work but **parallelisable** —
all pairwise scores computed at once. With enough hardware the transformer is
faster wall-clock despite more FLOPs.

**Very long sequences (Q1f).** For `T→∞`, attention's `O(T²)` memory/time
becomes prohibitive; an RNN's `O(T)` memory (constant per step) is preferable,
at the cost of sequential compute and long-range decay. (Modern answer:
linear-attention / state-space models trade these off.)

**Continuous inputs (Q2d).** Tokenise by chunking the signal (image patches,
audio frames), linearly embed each chunk, and add positional encodings that
respect the 1-D/2-D geometry; self-attention then models local + global
relations across chunks. This is exactly the ViT patchification.

**Unbounded positions (Q3b).** Replace the *learned* positional table with a
functional encoding (sinusoidal or RoPE) that is defined for any index, removing
the fixed maximum sequence length.

**CNN vs ViT (Q3e).** CNNs' strong locality bias helps on **small** datasets and
generalises across image sizes (translation equivariance); ViTs have weaker
priors, so they need **more data** but can model global structure and, with
learned positional encodings, are tied to the training resolution unless the
encoding is interpolated/extended.

**Generation cost (Q4e / KV cache).** Generating `T` tokens needs `T` forward
passes, each attending over the growing prefix — more expensive than one
training forward pass over the same length. **KV caching** stores past keys and
values so each new token only computes attention against cached states instead
of recomputing the whole prefix, turning per-step cost from `O(t²)` to `O(t)`.

**Nucleus sampling (Q4h).** Top-p sampling draws from the smallest set of tokens
whose cumulative probability exceeds `p`, discarding the unreliable tail. It
yields more diverse, less repetitive text than greedy decoding at the cost of
determinism (and a little quality variance).

---

## HW4 — Representation learning

**Augmentations → similarity groups (Q1).** With `K` images and augmentation
distribution `p_aug(·|img)`, set `N = K` and `p_i = p_aug(·|img_i)`: each image
defines one similarity group, its augmentations are the positives.

**Contrastive = classification (Q2).** Eq. (1) is a `C = N`-way cross entropy;
the logits are the dot products `f(x_i)ᵀf(y_j)` (bounded in `[−1, 1]` on the unit
sphere) and the target class is the positive index `i`.

**Distances vs dot products (Q3).** `‖u−v‖² = ‖u‖² + ‖v‖² − 2uᵀv`; on the unit
sphere `‖u−v‖² = 2 − 2·cos(u,v)`, so minimising distance ⇔ maximising cosine
similarity.

**Feature geometry (Q4).** Perfect contrastive accuracy is guaranteed by
condition (ii): *for every group, its diameter is smaller than the margin to
**all** other groups*. Optimal encoders therefore make each group a **tight
cluster** (small diameter → alignment/invariance to within-group variation) that
is **well-separated / uniformly spread** on the sphere (large margins →
uniformity, cf. Tammes' problem). This is the alignment+uniformity picture.

**Normalisation (Q5).** An encoder with perfect accuracy does **not** achieve
zero contrastive loss (the softmax cross entropy is > 0 unless logits are
±∞). Without the sphere constraint one can drive the loss arbitrarily low by
scaling features (temperature → 0 sharpens the softmax), but this causes
numerical-stability issues — hence l2-normalisation + a fixed temperature.

**XCE as dual-encoder contrastive (Q6).** Supervised cross entropy is exactly
dual-encoder contrastive with `N = C`, `X1 = X` (inputs), `X2 = {1..C}` (labels),
`f1(x) = [g_feature(x); 1]`, `f2(c) = W_c` (final-layer weights), and
`p_i = Uniform{x : y=i}`. Prediction correctness coincides between the two views.

**Reconstruction vs similarity (Q7–8).** The **autoencoder** objective only
requires invertibility on the training set: the theoretical optimum can encode
the dataset index and thus *need not* group by semantic factors like colour —
nearest neighbours in AE space are not guaranteed to share colour/shape.
**Contrastive** encoders are, by construction, invariant to whatever the
augmentations vary and sensitive to the rest:
- recolor augmentation → invariant to **colour**, sensitive to shape/location;
- translate augmentation → invariant to **location**;
- jitter → preserves all factors.

Our k-NN probes in `results/hw4/repr_probe.json` make this quantitative (e.g.
the recolor-augmented encoder drops colour-prediction accuracy toward chance
while keeping shape accuracy high).

---

## HW5 — Generative models

**ELBO terms (Q1).** The first term `E_q[log p_θ(x|z)]` is the **reconstruction**
likelihood (decoder should reconstruct `x` from `z`); the second
`−KL(q_φ(z|x)‖p(z))` is a **regulariser** pulling the posterior toward the prior
`N(0,I)`, keeping the latent space smooth/samplable.

**Means suffice? (Q2).** (a) No — decoding only the means does **not** recover
the data distribution in general; the per-sample variances carry information the
decoder relies on. (b) The means `f^μ(x)` are **not** Gaussian in general; the
*sum* `z = μ + σ·ε` being unit-Gaussian does not imply its mean component is.

**AE ≠ generative (Q3).** A perfectly reconstructing deterministic autoencoder
gives no guarantee that `g(z) ∼ p_data` for `z ∼ N(0,I)`: nothing constrains the
*distribution* of codes, only that training points round-trip.

**Forward process (Q7).** By adding independent Gaussians,
`q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I)` with `ᾱ_t = Π_{s≤t} α_s`. Implemented as
`q_sample` and used by the training loss.

**KL to prior (Q8).** `D_KL(q(x_T|x_0)‖N(0,I)) = ½[ᾱ_T‖x_0‖² −
d·log(1−ᾱ_T) − d·ᾱ_T]` → 0 as `ᾱ_T → 0` for large `T`; so `x_T` is
approximately unit Gaussian.

**Reverse posterior & ε-loss (Q9–11).** The tractable posterior
`q(x_{t−1}|x_t,x_0)` is Gaussian; re-parameterising the mean in terms of the
predicted noise gives
`μ_θ = (1/√α_t)(x_t − (1−α_t)/√(1−ᾱ_t)·ε_θ)`, and with `σ_t² = β̃_t` the KL
objective reduces to the simple noise-prediction loss
`L = ‖ε_t − ε_θ(x_t, t)‖²`. This is exactly `diffusion_loss`.

**VAE vs diffusion (Q14).** The diffusion "encoder" is the **fixed** forward
noising process (no learned parameters), its "latent" is the noisy `x_T`, and its
"decoder" is the learned reverse chain. Unlike a VAE, the diffusion encoder is
fixed/non-parametric and the latent has the same dimensionality as the data.
