"""Exact-match accuracy of a fine-tuned model on a Q&A split (chapters 2 and 3).

Generates an answer for every prompt (greedy, stops at <|endoftext|>) and compares it to the
expected completion after normalizing: lowercase, no trailing period, no leading "the".

Usage (Hub):
  python src/eval.py --model NAME/tinyllm-closed --data NAME/tinyllm-closed --split test
  python src/eval.py --model NAME/tinyllm-closed --data NAME/tinyllm-closed --split train --limit 300
Usage (local):
  python src/eval.py --model out/closed --data data/hf/tinyllm-closed --split test
Eval suite (every tier of build_eval_suite.py; results saved as JSON for comparing runs):
  python src/eval.py --model out/open --suite data/hf/tinyllm-eval --out results/open-v1.json
"""

import argparse
import collections
import json
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


def score(model, tok, ds, batch, dev):
    """Greedy answers for every prompt -> per-attribute hits/totals, memory answers, misses."""
    hits, totals, memory, wrong = collections.Counter(), collections.Counter(), 0, []
    for i in range(0, len(ds), batch):
        rows = ds[i: i + batch]
        enc = tok(rows["prompt"], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        mems = rows.get("memory") or [""] * len(answers)
        for prompt, got, want, attr, mem in zip(rows["prompt"], answers, rows["completion"],
                                                rows["attribute"], mems):
            ok = norm(got) == norm(want)
            hits[attr] += ok
            totals[attr] += 1
            memory += bool(mem) and norm(got) == norm(mem)  # counterfactual: answered from memory
            if not ok:
                wrong.append((prompt, want, got))
    return hits, totals, memory, wrong


def report(title, hits, totals, memory, wrong, show):
    n, h = sum(totals.values()), sum(hits.values())
    print(f"{title}: {h}/{n} = {h / n:.1%}" + (f"   (answered from memory: {memory / n:.1%})" if memory else ""))
    for attr in sorted(totals):
        print(f"  {attr:<10} {hits[attr]:>4}/{totals[attr]:<4} {hits[attr] / totals[attr]:.1%}")
    for prompt, want, got in wrong[:show]:
        q = prompt.split("<|q|>")[-1].replace("<|a|>", "").strip()
        print(f"  wrong: {q!r}  expected {want.strip()!r}  got {got.strip()!r}")
    return {"accuracy": h / n, "n": n, "memory": memory / n,
            "by_attribute": {a: hits[a] / totals[a] for a in sorted(totals)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hub id or local folder")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--data", help="Hub dataset id or local folder")
    what.add_argument("--suite", help="eval suite (tinyllm-eval): Hub id or local folder; runs every tier")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (per tier)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--show", type=int, default=8, help="print this many wrong answers (per tier)")
    ap.add_argument("--out", help="write the results here as JSON")
    args = ap.parse_args()

    source = args.suite or args.data
    dd = load_from_disk(source) if os.path.isdir(source) else load_dataset(source)
    splits = list(dd) if args.suite else [args.split]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # pad on the left so every prompt ends right where generation starts
    dev = device()
    model = AutoModelForCausalLM.from_pretrained(args.model).to(dev).eval()

    results = {}
    for split in splits:
        ds = dd[split]
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        results[split] = report(f"{args.model} on {source} [{split}]",
                                *score(model, tok, ds, args.batch, dev), args.show)

    if args.suite:
        print(f"\n{'tier':<15} accuracy")
        for split, r in results.items():
            print(f"{split:<15} {r['accuracy']:>7.1%}" + (f"   memory {r['memory']:.1%}" if r["memory"] else ""))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "data": source, "results": results}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
