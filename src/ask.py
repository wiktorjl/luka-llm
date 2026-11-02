"""Ask the closed-book model about a country: capital, currency, language or continent.

Wraps each question in the Q&A format it was fine-tuned on ("<|q|> ...<|a|>") and answers greedily.

Usage:
  python src/ask.py --model NAME/tinyllm-countries-closed "What is the capital of France?"
  python src/ask.py --model out/tinyllm-countries/closed            # interactive: one question per line
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hub id or local folder of the closed-book model")
    ap.add_argument("questions", nargs="*", help="questions to answer; none = read them from the keyboard")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    dev = device()
    model = AutoModelForCausalLM.from_pretrained(args.model).to(dev).eval()

    def answer(q):
        enc = tok(f"<|q|> {q.strip()}<|a|>", return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if args.questions:
        for q in args.questions:
            print(f"{q}  ->  {answer(q)}")
        return
    print("Ask about a country's capital, currency, language or continent (Ctrl-D to quit).")
    while True:
        try:
            q = input("> ")
        except EOFError:
            break
        if q.strip():
            print(answer(q))


if __name__ == "__main__":
    main()
