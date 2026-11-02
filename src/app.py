"""Web demo: one prompt, three models side by side (needs: pip install gradio).

Usage:
  python src/app.py --hf-user NAME     # models from the Hub (run `hf auth login` first)
  python src/app.py --local out/tinyllm-countries   # models trained locally
Then open the printed local URL. Add --share for a temporary public link.
On a Hugging Face Space: set variable TINYLLM_USER=NAME and secret HF_TOKEN; no arguments needed.
"""

import argparse
import os

import gradio as gr
from transformers import pipeline

EXAMPLES = [
    "The capital of Poland is",
    "<|q|> What is the capital of Poland?<|a|>",
    "<|q|> What is the currency of Japan?<|a|>",
    "<|q|> What language do people speak in Brazil?<|a|>",
    "<|q|> What is the capital of South Korea?<|a|>",
    "<|q|> What is the capital of Zorbania?<|a|>",
    "<|ctx|> Zorbania is a country in Europe. The capital of Zorbania is Plinktown.<|q|> What is the capital of Zorbania?<|a|>",
    "<|ctx|> Poland is a country in Europe. The capital of Poland is Kraków.<|q|> What is the capital of Poland?<|a|>",
]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hf-user", default=os.environ.get("TINYLLM_USER"),
                   help="Hub username; on a Space, set the TINYLLM_USER variable instead")
    g.add_argument("--local")
    ap.add_argument("--prefix", default="tinyllm-countries", help="Hub model names are NAME/PREFIX-base etc.")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    if not (args.hf_user or args.local):
        ap.error("pass --hf-user NAME or --local out (or set TINYLLM_USER)")

    names = ("base", "closed", "open")
    paths = {n: os.path.join(args.local, n) if args.local else f"{args.hf_user}/{args.prefix}-{n}" for n in names}
    models = {n: pipeline("text-generation", model=p) for n, p in paths.items()}

    def compare(prompt):
        # greedy, stops at <|endoftext|>; return_full_text=False keeps only the continuation
        return [models[n](prompt, max_new_tokens=60, do_sample=False,
                          return_full_text=False)[0]["generated_text"] for n in names]

    gr.Interface(
        fn=compare,
        inputs=gr.Textbox(label="Prompt", lines=4),
        outputs=[gr.Textbox(label=f"{args.prefix}-{n}") for n in names],
        examples=EXAMPLES,
        title="Tiny LLM: base vs closed-book vs open-book",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
