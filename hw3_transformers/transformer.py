"""HW3 -- a Transformer implemented from scratch in PyTorch.

Implements, per the assignment:
  * ``AttentionHead``      -- single-headed scaled dot-product self-attention
    with support for an additive attention mask (0 => "do not attend");
  * ``MultiHeadedAttention`` -- splits into heads, concatenates, projects with W_O;
  * ``FFN``                -- LayerNorm + Linear + GELU + Linear (pre-norm);
  * ``AttentionResidual``  -- residual around attention and around the FFN;
  * ``Transformer``        -- a stack of AttentionResidual blocks.

The building blocks are shared by the ViT (image classification) and DialogueGPT
(language modelling) tasks in the sibling modules.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.WQ = nn.Linear(d_model, d_k, bias=False)
        self.WK = nn.Linear(d_model, d_k, bias=False)
        self.WV = nn.Linear(d_model, d_k, bias=False)
        self.d_k = d_k

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        # x: (B, T, d_model)
        Q = self.WQ(x)                      # (B, T, d_k)
        K = self.WK(x)
        V = self.WV(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, T, T)
        if attn_mask is not None:
            # attn_mask in {0,1}; where 0 the token must not be attended to.
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        A = F.softmax(scores, dim=-1)
        return A @ V, A


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.d_k) for _ in range(n_heads)]
        )
        self.WO = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None):
        outs, attns = [], []
        for head in self.heads:
            o, a = head(x, attn_mask)
            outs.append(o)
            attns.append(a)
        concat = torch.cat(outs, dim=-1)          # (B, T, d_model)
        # stack head attentions: (B, n_heads, T, T)
        attn = torch.stack(attns, dim=1)
        return self.WO(concat), attn


class FFN(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class AttentionResidual(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadedAttention(d_model, n_heads)
        self.ffn = FFN(d_model, ffn_hidden)

    def forward(self, x, attn_mask=None):
        a, attn = self.attn(self.norm(x), attn_mask)
        x = x + a                     # residual after attention
        x = x + self.ffn(x)           # residual after FFN
        return x, attn


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ffn_hidden=None):
        super().__init__()
        ffn_hidden = ffn_hidden or 4 * d_model
        self.blocks = nn.ModuleList(
            [AttentionResidual(d_model, n_heads, ffn_hidden) for _ in range(n_layers)]
        )

    def forward(self, x, attn_mask=None, return_attn=False):
        attns = []
        for block in self.blocks:
            x, attn = block(x, attn_mask)
            attns.append(attn)
        if return_attn:
            return x, attns
        return x
