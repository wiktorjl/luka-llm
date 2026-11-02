"""Package our data for the preamble (tools) path.

Turns the Plan v1 artifacts into what transformers/TRL load by name:
  tokenizer.json            -> a transformers tokenizer folder (AutoTokenizer can load it)
  train_mix.txt, val_mix.txt -> a dataset with one {"text": document} row per document

Saves both under --out-dir. With --hf-user, also pushes them as private Hub repos
<user>/<prefix>-tokenizer and <user>/<prefix>-pretrain (run `hf auth login` first).

Usage:
  python export_for_trl.py --tokenizer data/tokenizer.json \
      --train data/train_mix.txt --val data/val_mix.txt --out-dir data/hf [--hf-user YOUR_NAME]

Needs: pip install transformers datasets
"""

import argparse
import os

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast

SEP = "<|endoftext|>"


def iter_docs(path, chunk_size=1 << 24):
    buf = ""
    with open(path, encoding="utf-8") as f:
        while chunk := f.read(chunk_size):
            buf += chunk
            parts = buf.split(SEP)
            buf = parts.pop()
            yield from (p.strip() for p in parts if p.strip())
    if buf.strip():
        yield buf.strip()


def gen(path):
    for doc in iter_docs(path):
        yield {"text": doc}  # TRL appends <|endoftext|> to every document itself


def build_tokenizer(path):
    return PreTrainedTokenizerFast(
        tokenizer_file=path,
        eos_token="<|endoftext|>",  # end of document / end of answer; TRL appends it
        pad_token="<|pad|>",
        additional_special_tokens=["<|q|>", "<|a|>", "<|ctx|>"],
        model_max_length=512,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-user", help="also push to the Hub as private repos under this account")
    ap.add_argument("--prefix", default="tinyllm-countries",
                    help="name prefix for repos and folders (tinyllm-countries keeps these apart from the TinyStories ones)")
    args = ap.parse_args()

    tok = build_tokenizer(args.tokenizer)
    tok_dir = os.path.join(args.out_dir, f"{args.prefix}-tokenizer")
    tok.save_pretrained(tok_dir)
    print(f"tokenizer: {len(tok)} tokens, eos={tok.eos_token_id}, pad={tok.pad_token_id} -> {tok_dir}")

    ds = DatasetDict({
        "train": Dataset.from_generator(gen, gen_kwargs={"path": args.train}),
        "validation": Dataset.from_generator(gen, gen_kwargs={"path": args.val}),
    })
    ds_dir = os.path.join(args.out_dir, f"{args.prefix}-pretrain")
    ds.save_to_disk(ds_dir)
    print(f"dataset: {ds['train'].num_rows:,} train / {ds['validation'].num_rows:,} validation rows -> {ds_dir}")

    if args.hf_user:
        tok.push_to_hub(f"{args.hf_user}/{args.prefix}-tokenizer", private=True)
        ds.push_to_hub(f"{args.hf_user}/{args.prefix}-pretrain", private=True)
        print(f"pushed {args.hf_user}/{args.prefix}-tokenizer and {args.hf_user}/{args.prefix}-pretrain")


if __name__ == "__main__":
    main()
