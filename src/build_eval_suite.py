"""Build the frozen open-book eval suite: five tiers, each a split of one dataset.

  in_dist         invented countries from the training generator (like open test, but 1,000 rows)
  no_shortcut     same, but language/currency names are NOT built from the country name
                  ("Nosporland" -> "Quelvan", not "Nosporish"), so only the card gives the answer
  counterfactual  real countries with one fact swapped ("The capital of France is Lyon"):
                  the right answer is the card's, not the memorized one (kept in `memory`)
  distractors     one card describing two invented countries, question about one of them
  paraphrase      card sentences and questions worded unlike anything in training

Tiers 2-5 all use decoupled names, so in_dist - no_shortcut measures how much the model guesses
from the name. Every invented name is new: none overlaps real countries or the training fakes.
Fixed seed: rebuild only when you mean to change the suite, and compare runs on the same version.

Usage:
  python src/build_eval_suite.py --countries data/countries_clean.jsonl \
      --fakes data/sft/fake_countries.jsonl --out-dir data/eval_suite --hf-out data/hf [--hf-user NAME]
"""

import argparse
import json
import os
import random

from build_sft_data import (ATTRS, CARD, QUESTIONS, RESERVED, answer, cap_first, fake_countries,
                            qa_rows, stem, write_jsonl)

VERSION = "v1"

# Wording never used in training (compare QUESTIONS and CARD in build_sft_data.py).
NEW_QUESTIONS = {
    "capital": ["Name the capital of {c}.", "Which city serves as the capital of {c}?"],
    "continent": ["Name the continent where {c} lies.", "{C} is on which continent?"],
    "language": ["Name the main language of {c}.", "Which language do the people of {c} use?"],
    "currency": ["Name the currency of {c}.", "Which currency is used in {c}?"],
}
NEW_CARD = {
    "continent": ["{C} lies in {v}.", "You can find {c} in {v}."],
    "capital": ["{C}'s capital city is {v}.", "The seat of government of {c} is {v}."],
    "language": ["The main language of {c} is {v}.", "{v} is spoken in {c}."],
    "currency": ["{C} uses the {v}.", "The currency of {c} is the {v}."],
}


def decouple(fakes, taken, rng):
    """Give each invented country a language (and currency) unrelated to its name."""
    for f in fakes:
        while True:
            lang = cap_first(stem(rng, 2) + rng.choice(["ish", "ese", "ian", "an", "ic"]))
            if lang.lower() not in taken and not lang.lower().startswith(f["name"][:3].lower()):
                break
        taken.add(lang.lower())
        f["language"] = lang
        f["currency"] = f"{lang} {f['currency'].split()[-1]}"
    return fakes


def sentences(rec, attrs, rng, templates):
    return [rng.choice(templates[a]).format(c=rec["display"], C=cap_first(rec["display"]), v=rec[a])
            for a in attrs]


def row(card, question, rec, attr, tier, memory=""):
    return {"prompt": f"<|ctx|> {card}<|q|> {question}<|a|>", "completion": answer(rec[attr]),
            "country": rec["name"], "attribute": attr, "tier": tier, "memory": memory}


def ask(rec, attr, rng, questions=QUESTIONS):
    return rng.choice(questions[attr]).format(c=rec["display"], C=cap_first(rec["display"]))


def counterfactual(real, rng):
    rows = []
    for rec in real:
        attrs = [a for a in ATTRS if rec.get(a)]
        for a in attrs:
            others = sorted({r[a] for r in real if r.get(a) and r[a] != rec[a]})
            fake = {**rec, a: rng.choice(others)}
            order = attrs[:]
            rng.shuffle(order)
            card = " ".join(sentences(fake, order, rng, CARD))
            rows.append(row(card, ask(fake, a, rng), fake, a, "counterfactual", memory=rec[a]))
    return rows


def distractors(fakes, rng):
    rows = []
    for target, other in zip(fakes[0::2], fakes[1::2]):
        for a in ATTRS:
            facts = sentences(target, ATTRS, rng, CARD) + sentences(other, ATTRS, rng, CARD)
            rng.shuffle(facts)
            rows.append(row(" ".join(facts), ask(target, a, rng), target, a, "distractors"))
    return rows


def paraphrase(fakes, rng):
    rows = []
    for rec in fakes:
        for a in ATTRS:
            order = ATTRS[:]
            rng.shuffle(order)
            card = " ".join(sentences(rec, order, rng, NEW_CARD))
            rows.append(row(card, ask(rec, a, rng, NEW_QUESTIONS), rec, a, "paraphrase"))
    return rows


def tagged(rows, tier):
    return [{**{k: v for k, v in r.items() if k != "fake"}, "tier": tier, "memory": ""} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", required=True, help="countries_clean.jsonl")
    ap.add_argument("--fakes", required=True, help="fake_countries.jsonl from build_sft_data.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-tier", type=int, default=250, help="invented countries per tier (x4 questions)")
    ap.add_argument("--hf-out", help="also save a Hugging Face dataset (one split per tier) here")
    ap.add_argument("--hf-user", help="also push it to the Hub as a private repo")
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.countries, encoding="utf-8") as f:
        real = [json.loads(line) for line in f]
    with open(args.fakes, encoding="utf-8") as f:
        seen = [json.loads(line) for line in f]

    taken = {r["name"] for r in real} | {v for f in seen for v in (f["name"], f["capital"], f["language"])}
    fakes = fake_countries(5 * args.per_tier, taken | RESERVED, rng)
    taken_lower = {x.lower() for x in taken | RESERVED} | {
        v.lower() for f in fakes for v in (f["name"], f["capital"], f["language"])}
    n = args.per_tier
    plain, rest = fakes[:n], decouple(fakes[n:], taken_lower, rng)

    tiers = {
        "in_dist": tagged([x for r in plain for x in qa_rows(r, rng, 1, True)], "in_dist"),
        "no_shortcut": tagged([x for r in rest[:n] for x in qa_rows(r, rng, 1, True)], "no_shortcut"),
        "counterfactual": counterfactual(real, rng),
        "distractors": distractors(rest[n: 3 * n], rng),
        "paraphrase": paraphrase(rest[3 * n: 4 * n], rng),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in tiers.items():
        write_jsonl(os.path.join(args.out_dir, f"{name}.jsonl"), rows)
        print(f"{name:<15} {len(rows):>5} rows   e.g. {rows[0]['prompt']}{rows[0]['completion']}")
    with open(os.path.join(args.out_dir, "VERSION"), "w") as f:
        f.write(f"{VERSION} seed={args.seed}\n")

    if args.hf_out or args.hf_user:
        from datasets import Dataset, DatasetDict
        dd = DatasetDict({name: Dataset.from_list(rows) for name, rows in tiers.items()})
        if args.hf_out:
            dd.save_to_disk(os.path.join(args.hf_out, "tinyllm-eval"))
        if args.hf_user:
            dd.push_to_hub(f"{args.hf_user}/tinyllm-eval", private=True)
        print("saved Hugging Face dataset" + (f", pushed to {args.hf_user}" if args.hf_user else ""))


if __name__ == "__main__":
    main()
