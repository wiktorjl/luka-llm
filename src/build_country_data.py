"""Render countries.json into short English prose for pretraining (Plan v1, Chapter 1, step 2).

Outputs (in --out-dir):
  countries_clean.jsonl  one normalized fact record per country; Q&A pairs and fact cards
                         (chapters 2 and 3) are generated from this same file later
  countries_prose.txt    short snippets separated by the same <|endoftext|> line used in
                         the TinyStoriesV2 files, so both corpora go through one tokenizer path

Usage:
  python build_country_data.py --countries data/countries.json --out-dir data/ \
      --stories data/tinystories/TinyStoriesV2-GPT4-train.txt --fraction 0.05
  (without --stories, --target-mb sets the output size directly)
"""

import argparse
import json
import os
import random
import string

SEP = "\n<|endoftext|>\n"  # TinyStoriesV2 story separator; confirm with `head` on your copy

# Names that read naturally with "the" in front.
ARTICLED = {
    "Bahamas", "Central African Republic", "Comoros", "Dominican Republic", "Gambia",
    "Maldives", "Marshall Islands", "Netherlands", "Philippines", "Solomon Islands",
    "United Arab Emirates", "United Kingdom", "United States",
}

# ---------------------------------------------------------------- normalize

def continent(c):
    """Region from the data, with the Americas split into the two continents kids learn."""
    if c["region"] == "Americas":
        return "South America" if c["subregion"] == "South America" else "North America"
    return c["region"]


def normalize(raw):
    """One clean record per sovereign country. Ambiguous attributes become None."""
    out = []
    for c in raw:
        if not c.get("independent"):
            continue
        name = c["name"]["common"]
        langs = list(c.get("languages", {}).values())   # order is by ISO code, NOT importance
        curs = [v["name"] for v in c.get("currencies", {}).values()]
        out.append({
            "name": name,
            "display": f"the {name}" if name in ARTICLED else name,
            "capital": c["capital"][0] if c.get("capital") else None,  # South Africa: Pretoria
            "continent": continent(c),
            # 1-2 languages: state them; 3+ (India, Zimbabwe...): too ambiguous to teach
            "language": " and ".join(langs) if 1 <= len(langs) <= 2 else None,
            "language_count": len(langs),
            # exactly one currency, or skip (Panama, Zimbabwe, Micronesia)
            "currency": curs[0] if len(curs) == 1 else None,
        })
    return sorted(out, key=lambda r: r["name"])

# ---------------------------------------------------------------- templates
# Fields: c = country mid-sentence, C = country at sentence start, cap, cont, lang,
# lang_is ("is"/"are"), cur, name/name2, pron/Pron/poss.

FACTS = {
    "capital": [
        "The capital of {c} is {cap}.",
        "{cap} is the capital of {c}.",
        "The capital city of {c} is {cap}.",
        "In {c}, the capital is {cap}.",
        "{cap} is a city in {c}. It is the capital.",
    ],
    "continent": [
        "{C} is a country in {cont}.",
        "{C} is in {cont}.",
        "You can find {c} in {cont}.",
        "{C} is one of the countries in {cont}.",
    ],
    "language": [
        "People in {c} speak {lang}.",
        "In {c}, people speak {lang}.",
        "{lang} {lang_is} spoken in {c}.",
        "The people of {c} talk in {lang}.",
    ],
    "currency": [
        "The money in {c} is the {cur}.",
        "In {c}, people pay with the {cur}.",
        "The {cur} is the money of {c}.",
        "People in {c} buy things with the {cur}.",
    ],
}

STORIES = [
    "{name} lived in {cap}, the capital of {c}. Every day, {pron} played in the park with {poss} friends.",
    "One day, {name} went on a trip to {c}. {C} is a country in {cont}. {Pron} saw the capital, {cap}, and {pron} was very happy.",
    "{name} looked at a big map with Mom. Mom pointed and said, \"This is {c}. It is in {cont}, and the capital is {cap}.\" {name} smiled.",
    "{name} had a new friend from {c}. The friend spoke {lang} at home. They played with a red ball all day.",
    "{name} found some coins in a box. \"These are from {c},\" said Dad. \"In {c}, people pay with the {cur}.\"",
    "{name} and {name2} played a guessing game. {name2} said, \"I am thinking of a country in {cont}. Its capital is {cap}.\" {name} said, \"It is {c}!\"",
    "{name} wanted to visit {cap}. Mom said, \"{cap} is the capital of {c}. People there speak {lang}.\" {name} clapped {poss} hands.",
]

KIDS = [("Lily", "she", "her"), ("Tim", "he", "his"), ("Ben", "he", "his"), ("Mia", "she", "her"),
        ("Sue", "she", "her"), ("Tom", "he", "his"), ("Anna", "she", "her"), ("Max", "he", "his"),
        ("Sam", "he", "his"), ("Lucy", "she", "her")]

FIELD_SOURCE = {"cap": "capital", "cont": "continent", "lang": "language", "cur": "currency"}


def fields_of(template):
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def usable(template, rec):
    """A template is usable if every fact it mentions exists for this country."""
    return all(rec[FIELD_SOURCE[f]] for f in fields_of(template) if f in FIELD_SOURCE)


def values(rec, rng):
    (name, pron, poss), name2 = rng.sample(KIDS, 2)
    disp = rec["display"]
    return {
        "c": disp, "C": disp[0].upper() + disp[1:],
        "cap": rec["capital"], "cont": rec["continent"], "cur": rec["currency"],
        "lang": rec["language"],
        "lang_is": "are" if rec["language"] and " and " in rec["language"] else "is",
        "name": name, "name2": name2[0], "pron": pron, "Pron": pron.capitalize(), "poss": poss,
    }


def snippet(rec, rng):
    """Either 2-4 plain fact sentences in random order, or one short story."""
    v = values(rec, rng)
    if rng.random() < 0.5:
        stories = [t for t in STORIES if usable(t, rec)]
        return rng.choice(stories).format(**v)
    attrs = [a for a in FACTS if rec[a]]
    rng.shuffle(attrs)
    picked = attrs[: rng.randint(min(2, len(attrs)), len(attrs))]
    return " ".join(rng.choice(FACTS[a]).format(**v) for a in picked)

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stories", help="TinyStories train file; sizes output as --fraction of the mix")
    ap.add_argument("--fraction", type=float, default=0.05, help="country share of the final mix")
    ap.add_argument("--target-mb", type=float, default=117.0, help="used when --stories is not given")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    with open(args.countries, encoding="utf-8") as f:
        recs = normalize(json.load(f))

    if args.stories:  # country / (stories + country) = fraction
        target = int(os.path.getsize(args.stories) * args.fraction / (1 - args.fraction))
    else:
        target = int(args.target_mb * 1_000_000)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "countries_clean.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rng = random.Random(args.seed)
    written = n = 0
    with open(os.path.join(args.out_dir, "countries_prose.txt"), "w", encoding="utf-8") as f:
        while written < target:
            s = snippet(rng.choice(recs), rng) + SEP
            f.write(s)
            written += len(s.encode("utf-8"))
            n += 1

    print(f"countries:            {len(recs)}")
    for a in ("capital", "continent", "language", "currency"):
        print(f"  with {a + ':':<11} {sum(1 for r in recs if r[a])}")
    print(f"snippets:             {n:,}  (~{n // len(recs):,} per country)")
    print(f"size:                 {written / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
