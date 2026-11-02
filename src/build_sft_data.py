"""Build the fine-tuning datasets for chapters 2 and 3 from countries_clean.jsonl.

Chapter 2, closed-book Q&A (the answer must come from what the model memorized):
  prompt:     "<|q|> What is the capital of Poland?<|a|>"
  completion: " Warsaw."
  train = ~90% of real countries; test = held-out real countries it never saw as Q&A
  (their facts WERE in pretraining).

Chapter 3, open-book Q&A (the answer is in a fact card in the prompt):
  prompt:     "<|ctx|> Zorbania is a country in Europe. The capital of Zorbania is Plinktown. ...<|q|> What is the capital of Zorbania?<|a|>"
  completion: " Plinktown."
  train = cards for real + invented countries; test = invented countries it never saw.
  Invented countries are the point: memory cannot answer them, so the model must read the card.

Never put <|endoftext|> in a completion: the trainer (TRL, or Plan v1's train.py) appends it.

Writes JSONL (for Plan v1) and, with --hf-out, Hugging Face datasets (for the preamble):
  closed_train.jsonl closed_test.jsonl open_train.jsonl open_test.jsonl fake_countries.jsonl

Usage:
  python build_sft_data.py --countries data/countries_clean.jsonl --out-dir data/sft \
      [--hf-out data/hf] [--hf-user YOUR_NAME]
"""

import argparse
import json
import os
import random

# Questions per attribute. {c} is the country as it reads mid-sentence ("the United States").
QUESTIONS = {
    "capital": ["What is the capital of {c}?", "Which city is the capital of {c}?",
                "What is the capital city of {c}?", "Tell me the capital of {c}."],
    "continent": ["Where is {c}?", "Which continent is {c} in?",
                  "On which continent is {c}?", "In what part of the world is {c}?"],
    "language": ["What language do people speak in {c}?", "Which language is spoken in {c}?",
                 "What do people speak in {c}?", "What language is used in {c}?"],
    "currency": ["What is the currency of {c}?", "What money do people use in {c}?",
                 "What do people in {c} pay with?", "What is the money of {c} called?"],
}

# Fact-card sentences: same wording family the model saw in pretraining (step 2).
CARD = {
    "continent": ["{C} is a country in {v}.", "{C} is in {v}."],
    "capital": ["The capital of {c} is {v}.", "{v} is the capital of {c}."],
    "language": ["People in {c} speak {v}.", "In {c}, people speak {v}."],
    "currency": ["The money in {c} is the {v}.", "In {c}, people pay with the {v}."],
}
ATTRS = list(QUESTIONS)
KEEP_IN_TRAIN = {"Poland"}  # demo country: always a trained question
RESERVED = {"Zorbania", "Plinktown", "Zorbish"}  # demo names: never generated, so the demo is unseen
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]


def answer(value):
    """' Warsaw.' / ' Washington D.C.' (no doubled period)."""
    return " " + value + ("" if value.endswith(".") else ".")


def cap_first(s):
    return s[0].upper() + s[1:]

# ---------------------------------------------------------------- invented countries

ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "k", "l", "m", "n", "p", "pl", "r", "s",
          "t", "tr", "v", "z"]
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["", "", "n", "r", "l", "s", "m"]


def stem(rng, syllables):
    """Short pronounceable stems ending in a consonant: 'Zorb', 'Kelvan', 'Drimos'."""
    s = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(syllables))
    return s if s[-1] not in "aeiou" else s + rng.choice(["n", "r", "l", "s"])


def fake_countries(n, real_names, rng):
    taken = {x.lower() for x in real_names | RESERVED}
    out = []
    while len(out) < n:
        s = stem(rng, rng.choice([1, 2, 2, 2]))
        names = {
            "name": cap_first(s + rng.choice(["ia", "land", "stan", "ania", "ova", "ica"])),
            "capital": cap_first(stem(rng, rng.choice([1, 2])) + rng.choice(["ton", "ville", "burg", "grad", "town"])),
            "language": cap_first(s + rng.choice(["ish", "ese", "ian"])),
        }
        if any(v.lower() in taken for v in names.values()):
            continue
        taken.update(v.lower() for v in names.values())
        out.append({
            **names,
            "display": names["name"],
            "continent": rng.choice(CONTINENTS),
            "currency": f"{names['language']} {rng.choice(['dollar', 'florin', 'crown', 'mark', 'coin', 'pound'])}",
            "fake": True,
        })
    return out

# ---------------------------------------------------------------- examples

def qa_rows(rec, rng, phrasings, with_card):
    """One row per (attribute, phrasing). Attributes the data leaves empty are skipped."""
    rows = []
    attrs = [a for a in ATTRS if rec.get(a)]
    for a in attrs:
        for q in rng.sample(QUESTIONS[a], phrasings):
            prompt = f"<|q|> {q.format(c=rec['display'])}<|a|>"
            if with_card:
                prompt = "<|ctx|> " + card(rec, attrs, rng) + prompt
            rows.append({"prompt": prompt, "completion": answer(rec[a]), "country": rec["name"],
                         "attribute": a, "fake": rec.get("fake", False)})
    return rows


def card(rec, attrs, rng):
    order = attrs[:]
    rng.shuffle(order)  # facts in random order, so position never gives the answer away
    d = rec["display"]
    return " ".join(rng.choice(CARD[a]).format(c=d, C=cap_first(d), v=rec[a]) for a in order)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", required=True, help="countries_clean.jsonl from step 2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout", type=float, default=0.10, help="share of real countries held out (chapter 2)")
    ap.add_argument("--fake-train", type=int, default=900)
    ap.add_argument("--fake-test", type=int, default=100)
    ap.add_argument("--hf-out", help="also save Hugging Face datasets here (preamble path)")
    ap.add_argument("--hf-user", help="also push them to the Hub as private repos")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.countries, encoding="utf-8") as f:
        real = [json.loads(line) for line in f]

    # Chapter 2: split real countries by country, so test questions are about unseen countries.
    candidates = [r for r in real if r["name"] not in KEEP_IN_TRAIN]
    held = set(r["name"] for r in rng.sample(candidates, round(len(real) * args.holdout)))
    closed_train = [x for r in real if r["name"] not in held for x in qa_rows(r, rng, 3, False)]
    closed_test = [x for r in real if r["name"] in held for x in qa_rows(r, rng, 4, False)]
    rng.shuffle(closed_train)

    # Chapter 3: cards for real countries + invented ones; test only on unseen invented ones.
    fakes = fake_countries(args.fake_train + args.fake_test, {r["name"] for r in real}, rng)
    open_train = [x for r in real + fakes[: args.fake_train] for x in qa_rows(r, rng, 1, True)]
    open_test = [x for r in fakes[args.fake_train:] for x in qa_rows(r, rng, 1, True)]
    rng.shuffle(open_train)

    os.makedirs(args.out_dir, exist_ok=True)
    files = {"closed_train": closed_train, "closed_test": closed_test,
             "open_train": open_train, "open_test": open_test}
    for name, rows in files.items():
        write_jsonl(os.path.join(args.out_dir, f"{name}.jsonl"), rows)
    write_jsonl(os.path.join(args.out_dir, "fake_countries.jsonl"), fakes)

    print(f"closed  train {len(closed_train):>5}  test {len(closed_test):>4}   "
          f"held-out countries ({len(held)}): {', '.join(sorted(held))}")
    print(f"open    train {len(open_train):>5}  test {len(open_test):>4}   "
          f"invented countries: {args.fake_train} train / {args.fake_test} test, e.g. "
          f"{', '.join(f['name'] for f in fakes[:4])}")

    if args.hf_out or args.hf_user:
        from datasets import Dataset, DatasetDict
        for kind in ("closed", "open"):
            dd = DatasetDict({s: Dataset.from_list(files[f"{kind}_{s}"]) for s in ("train", "test")})
            if args.hf_out:
                dd.save_to_disk(os.path.join(args.hf_out, f"tinyllm-{kind}"))
            if args.hf_user:
                dd.push_to_hub(f"{args.hf_user}/tinyllm-{kind}", private=True)
        print("saved Hugging Face datasets" + (f", pushed to {args.hf_user}" if args.hf_user else ""))


if __name__ == "__main__":
    main()
