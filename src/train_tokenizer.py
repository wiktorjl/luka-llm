"""Train the tokenizer (Plan v1, Chapter 1, step 4).

Learns a byte-level BPE vocabulary of 8,192 tokens from train_mix.txt and saves it as
tokenizer.json. Special tokens are reserved now, so fine-tuning never has to resize the model:

  <|endoftext|>  end of a document (pretraining) and end of an answer (fine-tuning)
  <|pad|>        filler for short sequences in a fine-tuning batch
  <|q|> <|a|>    question / answer markers (chapter 2)
  <|ctx|>        fact-card marker (chapter 3)

Usage:
  python train_tokenizer.py --train data/train_mix.txt --val data/val_mix.txt --out data/tokenizer.json

Needs: pip install tokenizers
"""

import argparse
import os
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SEP = "<|endoftext|>"
SPECIALS = ["<|endoftext|>", "<|pad|>", "<|q|>", "<|a|>", "<|ctx|>"]  # get IDs 0-4

EXAMPLES = [
    "Once upon a time, Lily went to the park with her dog.",
    "The capital of Poland is Warsaw.",
    "In Poland, people pay with the Polish złoty.",
    "Ouagadougou is the capital of Burkina Faso.",
    # Chapter 2 format: no space BEFORE a special token (it would become a wasted ' ' token)
    "<|q|> What is the capital of Poland?<|a|> Warsaw.<|endoftext|>",
]


def iter_docs(path, chunk_size=1 << 24):
    """Yield documents split on <|endoftext|> without loading the whole file."""
    buf = ""
    with open(path, encoding="utf-8") as f:
        while chunk := f.read(chunk_size):
            buf += chunk
            parts = buf.split(SEP)
            buf = parts.pop()
            yield from (p.strip() for p in parts if p.strip())
    if buf.strip():
        yield buf.strip()


def batches(path, max_bytes, size=1000):
    """Feed the trainer lists of documents. Separators are never shown to it as text."""
    batch, seen = [], 0
    for doc in iter_docs(path):
        batch.append(doc)
        seen += len(doc)
        if len(batch) == size:
            yield batch
            batch = []
        if max_bytes and seen >= max_bytes:
            break
    if batch:
        yield batch


def build(vocab_size):
    tok = Tokenizer(models.BPE())  # byte-level, so every input is encodable: no <unk> needed
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # GPT-2 style word split
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes are always tokens
        show_progress=True,
    )
    return tok, trainer


def report(tok, train_path, val_path):
    print(f"\nvocab size: {tok.get_vocab_size()}")
    print("special IDs:", {s: tok.token_to_id(s) for s in SPECIALS})

    val = list(iter_docs(val_path))
    encs = tok.encode_batch(val)
    n_tok = sum(len(e.ids) for e in encs)
    n_bytes = sum(len(d.encode("utf-8")) for d in val)
    bad = sum(tok.decode(e.ids) != d for e, d in zip(encs, val))
    print(f"val: {len(val):,} docs, {n_tok:,} tokens, {n_bytes / n_tok:.2f} bytes/token, "
          f"round-trip mismatches: {bad}")
    est = os.path.getsize(train_path) / (n_bytes / n_tok)
    print(f"train estimate: ~{est / 1e6:,.0f}M tokens")

    print("\nhow text splits into tokens:")
    for s in EXAMPLES:
        ids = tok.encode(s).ids
        print(f"  {len(ids):>2} tokens  {[tok.decode([i], skip_special_tokens=False) for i in ids]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-mb", type=float, default=0, help="train on the first N MB only (0 = all)")
    args = ap.parse_args()

    tok, trainer = build(args.vocab_size)
    t0 = time.time()
    tok.train_from_iterator(batches(args.train, int(args.max_mb * 1e6)), trainer=trainer)
    print(f"trained in {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"saved {args.out}")
    report(tok, args.train, args.val)


if __name__ == "__main__":
    main()
