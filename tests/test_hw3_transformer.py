"""Sanity + correctness tests for the from-scratch Transformer stack (HW3)."""
import torch

from hw3_transformers.transformer import (
    AttentionHead, MultiHeadedAttention, Transformer,
)
from hw3_transformers.vit import PatchEmbed, VisionTransformer
from hw3_transformers.dialogue_gpt import MyTokenizer, DialogueGPT, dialogue_loss


def test_attention_output_shapes():
    B, T, d = 2, 5, 16
    x = torch.randn(B, T, d)
    head = AttentionHead(d, 8)
    out, A = head(x)
    assert out.shape == (B, T, 8)
    assert A.shape == (B, T, T)
    assert torch.allclose(A.sum(-1), torch.ones(B, T), atol=1e-5)


def test_attention_mask_blocks_future():
    # test case 3 analogue: each token attends only to itself and the one before.
    B, T, d = 1, 4, 8
    x = torch.randn(B, T, d)
    head = AttentionHead(d, d)
    mask = torch.zeros(T, T)
    for i in range(T):
        mask[i, i] = 1
        if i > 0:
            mask[i, i - 1] = 1
    mask = mask.unsqueeze(0)
    _, A = head(x, attn_mask=mask)
    # attention weight must be zero everywhere the mask is zero.
    assert torch.all(A[mask.expand_as(A) == 0] < 1e-6)


def test_multihead_and_transformer_shapes():
    B, T, d = 2, 6, 32
    x = torch.randn(B, T, d)
    mha = MultiHeadedAttention(d, n_heads=4)
    out, attn = mha(x)
    assert out.shape == (B, T, d)
    assert attn.shape == (B, 4, T, T)
    tr = Transformer(d, n_heads=4, n_layers=2)
    assert tr(x).shape == (B, T, d)


def test_patch_embed_count():
    pe = PatchEmbed(img_size=32, patch_size=4, in_chans=3, embed_dim=128)
    x = torch.randn(2, 3, 32, 32)
    out = pe(x)
    assert out.shape == (2, 64, 128)   # 8x8 = 64 patches


def test_vit_forward():
    vit = VisionTransformer(n_layers=2, n_heads=4, embed_dim=64)
    x = torch.randn(3, 3, 32, 32)
    logits = vit(x)
    assert logits.shape == (3, 10)
    heat = vit.class_token_attention(x)
    assert heat.shape[0] == 3 and heat.shape[1] == heat.shape[2]


def test_tokenizer_roundtrip_and_start():
    tok = MyTokenizer(["To be or not to be", "that is the question"])
    ids = tok.encode("to be")
    assert ids[0] == tok.start_id
    assert tok.decode(ids) == "to be"


def test_dialogue_gpt_loss_and_generate():
    tok = MyTokenizer(["hello world", "good morning world"])
    gpt = DialogueGPT(tok.vocab_size, d_model=32, n_heads=2, n_layers=2,
                      max_len=16, pad_id=tok.pad_id)
    ids = torch.tensor([tok.encode("hello world") + [tok.pad_id]])
    logits = gpt(ids)
    assert logits.shape[-1] == tok.vocab_size
    loss = dialogue_loss(logits, ids, tok.pad_id)
    assert loss.item() > 0
    out = gpt.generate(tok.encode("hello"), num_tokens=5)
    assert len(out) == len(tok.encode("hello")) + 5
