"""Build the pretraining mix (Plan v1, Chapter 1, step 3).

Reads TinyStories (train + valid) and countries_prose.txt from step 2, cleans every
document, removes duplicate stories, shuffles at the document level, and writes:

  train_mix.txt   stories + country snippets, shuffled, <|endoftext|>-separated
  val_mix.txt     TinyStories' own validation stories + the same share of country snippets

Usage:
  python build_mix.py \
      --stories data/tinystories/TinyStoriesV2-GPT4-train.txt \
      --val-stories data/tinystories/TinyStoriesV2-GPT4-valid.txt \
      --countries data/countries_prose.txt --out-dir data/

Needs: pip install ftfy. Measured peak RAM is ~1.1x the stories file plus ~0.2 GB
(~3 GB for the V2 train file); runtime ~1-3 minutes.
"""

import argparse
import os
import random

import ftfy

SEP = "<|endoftext|>"
MIN_CHARS = 20  # drop empty and near-empty documents

# Fast path: straighten typographic punctuation so "don't" and "don’t" become one token pattern.
PUNCT = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
})
# Characters that appear when UTF-8 text was mis-decoded as Windows-1252 ("â€™" for "’").
MOJIBAKE_HINTS = ("â", "Ã", "Â")


def iter_docs(path, chunk_size=1 << 24):
    """Yield documents split on <|endoftext|>, wherever it appears, without loading the whole file."""
    buf = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while chunk := f.read(chunk_size):
            buf += chunk
            parts = buf.split(SEP)
            buf = parts.pop()  # last piece may be cut mid-document; keep it for the next chunk
            yield from parts
    yield buf


def clean(doc, stats):
    doc = doc.strip()
    if " \n" in doc:  # trailing spaces at line ends cost tokens and teach nothing
        doc = "\n".join(line.rstrip() for line in doc.split("\n"))
    while "  " in doc:  # collapse double spaces left after sentences
        doc = doc.replace("  ", " ")
    if not doc.isascii():
        if any(h in doc for h in MOJIBAKE_HINTS):
            doc = ftfy.fix_text(doc)  # slow (~1 MB/s), so only on suspected damage
            stats["ftfy_fixed"] += 1
        doc = doc.translate(PUNCT)
    return doc


def load(path, stats, dedupe):
    seen, docs = set(), []
    for raw in iter_docs(path):
        doc = clean(raw, stats)
        if len(doc) < MIN_CHARS:
            stats["dropped_short"] += 1
            continue
        if dedupe:
            if doc in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(doc)
        docs.append(doc)
    return docs


def write(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d)
            f.write("\n" + SEP + "\n")


def mb(docs):
    return sum(len(d.encode("utf-8")) + len(SEP) + 2 for d in docs) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", required=True)
    ap.add_argument("--val-stories", help="TinyStories valid file; without it, 1%% of train is held out")
    ap.add_argument("--countries", required=True, help="countries_prose.txt from step 2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    stats = {"ftfy_fixed": 0, "dropped_short": 0, "dropped_duplicate": 0}
    stories = load(args.stories, stats, dedupe=True)
    if args.val_stories:
        train_set = set(stories)
        val_all = load(args.val_stories, stats, dedupe=True)
        val_stories = [d for d in val_all if d not in train_set]  # no test leaking into training
        stats["val_also_in_train"] = len(val_all) - len(val_stories)
        del train_set, val_all
    else:
        rng.shuffle(stories)
        cut = len(stories) // 100
        val_stories, stories = stories[:cut], stories[cut:]

    # Country snippets repeat facts on purpose, so no dedupe. Give val the same country share as train.
    countries = load(args.countries, stats, dedupe=False)
    rng.shuffle(countries)
    n_val_c = round(len(countries) * len(val_stories) / max(len(stories), 1))
    val_c, train_c = countries[:n_val_c], countries[n_val_c:]

    train, val = stories + train_c, val_stories + val_c
    rng.shuffle(train)  # spread country facts evenly through the single training pass
    rng.shuffle(val)

    os.makedirs(args.out_dir, exist_ok=True)
    write(os.path.join(args.out_dir, "train_mix.txt"), train)
    write(os.path.join(args.out_dir, "val_mix.txt"), val)

    for name, s, c in (("train", stories, train_c), ("val", val_stories, val_c)):
        total = mb(s) + mb(c)
        share = mb(c) / total * 100 if total else 0.0
        print(f"{name:5}  stories {len(s):>9,}  country {len(c):>9,}  "
              f"{total:8.1f} MB  country share {share:.1f}% of bytes")
    print("cleaning:", ", ".join(f"{k} {v:,}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
