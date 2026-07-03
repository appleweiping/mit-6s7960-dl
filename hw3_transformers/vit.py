"""HW3 -- Vision Transformer (ViT) built on the from-scratch Transformer.

  * ``PatchEmbed`` splits an image into non-overlapping square patches and
    linearly projects each (implemented efficiently with a strided Conv2d, as
    the assignment hints);
  * ``VisionTransformer`` prepends a learned class token, adds a learned
    positional embedding, runs the Transformer, and classifies from the final
    class-token representation.

Also exposes ``class_token_attention`` to produce the class-token attention
heatmap (averaged over heads and layers) requested by the assignment.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hw3_transformers.transformer import Transformer


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        # A conv with kernel=stride=patch_size performs the flatten+linear-project.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, n_patches, embed_dim)
        x = self.proj(x)                    # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)    # (B, n_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128,
                 n_heads=4, n_layers=4, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.transformer = Transformer(embed_dim, n_heads, n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                   # (B, N+1, D)
        x = x + self.pos_embed
        if return_attn:
            x, attns = self.transformer(x, return_attn=True)
        else:
            x = self.transformer(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])                      # class token
        if return_attn:
            return logits, attns
        return logits

    @torch.no_grad()
    def class_token_attention(self, x):
        """Return the class-token attention over patches, averaged over all
        heads and layers, reshaped to a square grid (per image)."""
        _, attns = self.forward(x, return_attn=True)
        # each attn: (B, n_heads, T, T); take row 0 (class token) over patches 1:
        stack = torch.stack([a[:, :, 0, 1:].mean(dim=1) for a in attns], dim=0)
        avg = stack.mean(dim=0)                          # (B, n_patches)
        g = int(avg.shape[1] ** 0.5)
        return avg.reshape(avg.shape[0], g, g)
