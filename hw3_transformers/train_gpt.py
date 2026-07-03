"""HW3 -- train DialogueGPT on tiny-shakespeare and sample generations.

Reproduces the assignment's language-model task: build the tokenizer, train the
GPT with next-token cross entropy, and print generations after each epoch to
show that samples approach (broken but recognisable) English.

Outputs: results/hw3/gpt_metrics.json, results/hw3/gpt_samples.txt,
         results/hw3/gpt_loss.png
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from common.utils import set_seed
from hw3_transformers.data import shakespeare_lines
from hw3_transformers.dialogue_gpt import MyTokenizer, DialogueGPT, dialogue_loss


class DialogueDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len):
        self.examples = []
        for ln in lines:
            ids = tokenizer.encode(ln)[:max_len]
            if len(ids) >= 3:
                self.examples.append(torch.tensor(ids))
        self.pad_id = tokenizer.pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collate(pad_id):
    def collate(batch):
        return pad_sequence(batch, batch_first=True, padding_value=pad_id)
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--max-lines", type=int, default=6000)
    ap.add_argument("--outdir", default="results/hw3")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(0)

    lines = shakespeare_lines(min_len=20, max_lines=args.max_lines)
    tok = MyTokenizer(lines, max_vocab=6000)
    ds = DialogueDataset(lines, tok, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(tok.pad_id))
    print(f"vocab={tok.vocab_size}  examples={len(ds)}")

    model = DialogueGPT(tok.vocab_size, d_model=192, n_heads=6, n_layers=4,
                        max_len=args.max_len, pad_id=tok.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    losses, samples = [], []
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        for batch in loader:
            opt.zero_grad()
            logits = model(batch)
            loss = dialogue_loss(logits, batch, tok.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        avg = ep_loss / nb
        losses.append(avg)
        gen = model.generate(tok.encode("to be or"), num_tokens=30)
        text = tok.decode(gen)
        samples.append(f"[epoch {ep + 1}] {text}")
        print(f"epoch {ep + 1}/{args.epochs}  loss={avg:.3f}")
        print(f"  sample: {text}")
    elapsed = time.time() - t0

    # a longer 50-token generation as requested
    final_gen = tok.decode(model.generate(tok.encode("my lord"), num_tokens=50))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, "o-")
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss (next-token CE)")
    ax.set_title("HW3 DialogueGPT training loss (tiny-shakespeare)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "gpt_loss.png"), dpi=110)

    with open(os.path.join(args.outdir, "gpt_samples.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(samples))
        f.write(f"\n\n[final 50-token generation, prefix 'my lord']\n{final_gen}\n")

    metrics = {
        "epochs": args.epochs, "vocab_size": tok.vocab_size,
        "num_examples": len(ds), "final_loss": losses[-1],
        "loss_curve": losses, "seconds": round(elapsed, 1),
        "final_generation": final_gen,
    }
    with open(os.path.join(args.outdir, "gpt_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"final_loss": losses[-1], "seconds": metrics["seconds"]}, indent=2))
    print(f"final generation: {final_gen}")


if __name__ == "__main__":
    main()
