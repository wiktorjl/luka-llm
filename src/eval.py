"""Exact-match accuracy of a fine-tuned model on a Q&A split (chapters 2 and 3).

Generates an answer for every prompt (greedy, stops at <|endoftext|>) and compares it to the
expected completion after normalizing: lowercase, no trailing period, no leading "the".

Usage (Hub):
  python src/eval.py --model NAME/tinyllm-closed --data NAME/tinyllm-closed --split test
  python src/eval.py --model NAME/tinyllm-closed --data NAME/tinyllm-closed --split train --limit 300
Usage (local):
  python src/eval.py --model out/closed --data data/hf/tinyllm-closed --split test
"""

import argparse
import collections
import os

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def norm(s):
    s = s.strip().lower().rstrip(".")
    return s[4:] if s.startswith("the ") else s


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hub id or local folder")
    ap.add_argument("--data", required=True, help="Hub dataset id or local folder")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--show", type=int, default=8, help="print this many wrong answers")
    args = ap.parse_args()

    ds = (load_from_disk(args.data) if os.path.isdir(args.data) else load_dataset(args.data))[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # pad on the left so every prompt ends right where generation starts
    dev = device()
    model = AutoModelForCausalLM.from_pretrained(args.model).to(dev).eval()

    hits, totals, wrong = collections.Counter(), collections.Counter(), []
    for i in range(0, len(ds), args.batch):
        rows = ds[i: i + args.batch]
        enc = tok(rows["prompt"], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for prompt, got, want, attr in zip(rows["prompt"], answers, rows["completion"], rows["attribute"]):
            ok = norm(got) == norm(want)
            hits[attr] += ok
            totals[attr] += 1
            if not ok:
                wrong.append((prompt, want, got))

    n, h = sum(totals.values()), sum(hits.values())
    print(f"{args.model} on {args.data} [{args.split}]: {h}/{n} = {h / n:.1%}")
    for attr in sorted(totals):
        print(f"  {attr:<10} {hits[attr]:>4}/{totals[attr]:<4} {hits[attr] / totals[attr]:.1%}")
    for prompt, want, got in wrong[: args.show]:
        q = prompt.split("<|q|>")[-1].replace("<|a|>", "").strip()
        print(f"  wrong: {q!r}  expected {want.strip()!r}  got {got.strip()!r}")


if __name__ == "__main__":
    main()
