"""HW4 -- synthetic colored-shapes dataset (three factors: shape, location, color).

Matches the assignment's setup: 64x64 RGB images, each with an independently
sampled shape (square/circle/triangle), location, and color. Because the factors
are known and controllable, we can probe exactly which factors an encoder's
representation captures (via nearest neighbours), which is the whole point of the
reconstruction-vs-similarity comparison.
"""
from __future__ import annotations

import numpy as np

SHAPES = ["square", "circle", "triangle"]
COLORS = {
    "red": (1.0, 0.1, 0.1),
    "green": (0.1, 0.9, 0.2),
    "blue": (0.2, 0.3, 1.0),
    "yellow": (1.0, 0.9, 0.1),
}
COLOR_NAMES = list(COLORS.keys())
IMG = 64
HALF = 8  # half-size of a shape


def _draw(img, shape, cx, cy, color):
    c = np.array(color, dtype=np.float32)
    ys, xs = np.mgrid[0:IMG, 0:IMG]
    if shape == "square":
        mask = (np.abs(xs - cx) <= HALF) & (np.abs(ys - cy) <= HALF)
    elif shape == "circle":
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= HALF ** 2
    else:  # triangle (pointing up)
        dy = cy + HALF - ys
        mask = (dy >= 0) & (dy <= 2 * HALF) & (np.abs(xs - cx) <= (dy / 2))
    for k in range(3):
        img[k][mask] = c[k]
    return img


def sample_image(rng):
    shape = rng.choice(SHAPES)
    color = rng.choice(COLOR_NAMES)
    cx = int(rng.integers(HALF + 2, IMG - HALF - 2))
    cy = int(rng.integers(HALF + 2, IMG - HALF - 2))
    img = np.zeros((3, IMG, IMG), dtype=np.float32)  # black background
    _draw(img, shape, cx, cy, COLORS[color])
    factors = {"shape": SHAPES.index(shape), "color": COLOR_NAMES.index(color),
               "cx": cx, "cy": cy}
    return img, factors


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.empty((n, 3, IMG, IMG), dtype=np.float32)
    shapes = np.empty(n, dtype=np.int64)
    colors = np.empty(n, dtype=np.int64)
    locs = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        img, f = sample_image(rng)
        imgs[i] = img
        shapes[i] = f["shape"]
        colors[i] = f["color"]
        locs[i] = (f["cx"], f["cy"])
    return imgs, {"shape": shapes, "color": colors, "loc": locs}


# --- augmentations used to define contrastive positive pairs ---------------
def aug_recolor(img, rng):
    """Randomly recolor the shape -> encoder becomes INVARIANT to color."""
    out = img.copy()
    mask = out.sum(0) > 0.05
    new = np.array(COLORS[rng.choice(COLOR_NAMES)], dtype=np.float32)
    for k in range(3):
        out[k][mask] = new[k]
    return out


def aug_translate(img, rng, max_shift=10):
    """Random translation -> encoder becomes INVARIANT to location."""
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(np.roll(img, dx, axis=2), dy, axis=1)


def aug_jitter(img, rng):
    """Small brightness jitter -> mostly preserves all factors."""
    return np.clip(img * rng.uniform(0.85, 1.15), 0, 1).astype(np.float32)
