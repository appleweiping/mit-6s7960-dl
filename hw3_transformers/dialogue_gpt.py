"""HW3 -- DialogueGPT: a small GPT-style language model on the from-scratch Transformer.

Implements the assignment's tokenizer and model:
  * ``MyTokenizer``  -- word-level tokenizer with <start>, <pad>, <unk> special
    tokens; encode() prepends <start>, decode() joins with spaces;
  * ``DialogueGPT``  -- token + learned positional embeddings, a causal mask,
    the Transformer, and a language-model head; ``generate`` does greedy
    autoregressive decoding;
  * ``dialogue_loss`` -- next-token cross entropy, ignoring padding, not
    supervising the <start> token nor using the last logit.

Trained on the tiny-shakespeare corpus, split into lines as "dialogue" examples.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from hw3_transformers.transformer import Transformer


_WORD_RE = re.compile(r"\w+|[^\w\s]")


def word_tokenize(text: str):
    """A lightweight stand-in for nltk.word_tokenize (words + punctuation)."""
    return _WORD_RE.findall(text)


class MyTokenizer:
    START, PAD, UNK = "<start>", "<pad>", "<unk>"

    def __init__(self, texts, max_vocab=8000):
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(word_tokenize(t.lower()))
        vocab = [self.PAD, self.START, self.UNK]
        vocab += [w for w, _ in counter.most_common(max_vocab)]
        self.itos = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.pad_id = self.stoi[self.PAD]
        self.start_id = self.stoi[self.START]
        self.unk_id = self.stoi[self.UNK]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s: str):
        ids = [self.start_id]
        for w in word_tokenize(s.lower()):
            ids.append(self.stoi.get(w, self.unk_id))
        return ids

    def decode(self, ids):
        toks = [self.itos[i] for i in ids
                if i not in (self.pad_id, self.start_id)]
        return " ".join(toks)


class DialogueGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3,
                 max_len=64, pad_id=0):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = Transformer(d_model, n_heads, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self.pad_id = pad_id

    def _causal_mask(self, T, device):
        # lower-triangular {0,1}: token i attends to j<=i only.
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.tok_embed(input_ids) + self.pos_embed[:, :T]
        mask = self._causal_mask(T, input_ids.device)
        x = self.transformer(x, attn_mask=mask)
        x = self.norm(x)
        return self.lm_head(x)                     # (B, T, V)

    @torch.no_grad()
    def generate(self, prefix_ids, num_tokens, device="cpu"):
        ids = list(prefix_ids)
        for _ in range(num_tokens):
            inp = torch.tensor([ids[-self.max_len:]], device=device)
            logits = self.forward(inp)
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
        return ids


def dialogue_loss(logits, input_ids, pad_id):
    """Next-token cross entropy: predict token i+1 from logit i.

    Do not use the last logit; do not supervise the <start> position's target
    being <start>; ignore pad targets.
    """
    # shift: logits[:, :-1] predict input_ids[:, 1:]
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    target = input_ids[:, 1:].reshape(-1)
    return F.cross_entropy(pred, target, ignore_index=pad_id)
