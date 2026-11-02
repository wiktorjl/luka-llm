"""Run the demo prompts against base / closed / open side by side, or show chapter 1's learning curve.

Usage:
  python src/demo.py --hf-user NAME                 # all demo prompts x all three models
  python src/demo.py --local out/tinyllm-countries  # same, with models trained locally
  python src/demo.py --hf-user NAME --progression   # a fact completion from random weights to the final model
"""

import argparse
import glob
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEMOS = [
    ("1  fact completion",       "The capital of Poland is"),
    ("1  currency completion",   "In Japan, people pay with the"),
    ("2  trained question",      "<|q|> What is the capital of Poland?<|a|>"),
    ("2  currency question",     "<|q|> What is the currency of Poland?<|a|>"),
    ("2  language question",     "<|q|> What language do people speak in Poland?<|a|>"),
    ("2  held-out country",      "<|q|> What is the capital of South Korea?<|a|>"),
    ("2  country that isn't real", "<|q|> What is the capital of Zorbania?<|a|>"),
    ("3  answer from the card",  "<|ctx|> Zorbania is a country in Europe. The capital of Zorbania is Plinktown. "
                                 "People in Zorbania speak Zorbish.<|q|> What is the capital of Zorbania?<|a|>"),
    ("3  card beats memory",     "<|ctx|> Poland is a country in Europe. The capital of Poland is Kraków."
                                 "<|q|> What is the capital of Poland?<|a|>"),
]


PROGRESSION_PROMPT = "The capital of Poland is"


def generate(model, tok, prompt, max_new_tokens=40, sample=False):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=sample,
                             temperature=0.8 if sample else None, top_k=50 if sample else None,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def side_by_side(paths):
    tok = AutoTokenizer.from_pretrained(paths["base"])
    models = {n: AutoModelForCausalLM.from_pretrained(p).eval() for n, p in paths.items()}
    for label, prompt in DEMOS:
        print(f"\n### Chapter {label}\nprompt: {prompt}")
        for n, m in models.items():
            print(f"  {n:<7} -> {generate(m, tok, prompt)!r}")


def pick_checkpoints(commits):
    """commits oldest first -> checkpoints at doubling steps (1k, 2k, 4k, 8k, 16k), then the newest commit
    (the final model). Doubling, because most of the visible learning happens early."""
    steps = [c for c in commits if "step" in c.title.lower()]
    picked, i = [], 0
    while i < len(steps):
        picked.append((steps[i].title, steps[i].commit_id))
        i = 2 * i + 1
    return picked + [("final model", commits[-1].commit_id)]


def progression(base, hub):
    """Random weights, then each saved checkpoint, then the final model."""
    tok = AutoTokenizer.from_pretrained(base)
    stages = [("random weights", None)]
    if hub:
        from huggingface_hub import HfApi
        for title, rev in pick_checkpoints(list(reversed(HfApi().list_repo_commits(base)))):
            stages.append((title, dict(pretrained_model_name_or_path=base, revision=rev)))
    else:
        ckpts = sorted(glob.glob(os.path.join(base, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
        stages += [(os.path.basename(c), dict(pretrained_model_name_or_path=c)) for c in ckpts]
        stages += [("final", dict(pretrained_model_name_or_path=base))]
    for title, src in stages:
        torch.manual_seed(0)
        model = (AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(base)) if src is None
                 else AutoModelForCausalLM.from_pretrained(**src)).eval()
        print(f"\n[{title}]\n  {PROGRESSION_PROMPT} {generate(model, tok, PROGRESSION_PROMPT, 30, sample=True)}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf-user", help="load NAME/PREFIX-base, -closed, -open from the Hub")
    g.add_argument("--local", help="folder holding base/, closed/, open/ (out/tinyllm-countries)")
    ap.add_argument("--progression", action="store_true", help="chapter 1 learning curve from checkpoints")
    ap.add_argument("--prefix", default="tinyllm-countries", help="Hub model names are NAME/PREFIX-base etc.")
    args = ap.parse_args()

    if args.hf_user:
        paths = {n: f"{args.hf_user}/{args.prefix}-{n}" for n in ("base", "closed", "open")}
    else:
        paths = {n: os.path.join(args.local, n) for n in ("base", "closed", "open")
                 if os.path.isdir(os.path.join(args.local, n))}
    if args.progression:
        progression(paths["base"], hub=bool(args.hf_user))
    else:
        side_by_side(paths)


if __name__ == "__main__":
    main()